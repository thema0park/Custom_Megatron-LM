from operator import itemgetter

import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Module, Parameter
from .quant_utils import *


class QuantLinear(Module):
    """
    Class to quantize weights of given linear layer
    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 weight_bit=4,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode='symmetric',
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=0):
        super(QuantLinear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.counter = 0

        self.is_classifier = False

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={}, quantize_fn={})".format(
            self.weight_bit, self.full_precision_flag, self.quant_mode)
        return s

    def set_param(self, linear, model_dict=None, dict_idx=None):
        #self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        if model_dict is not None :
            self.weight = Parameter(model_dict[dict_idx+'.weight'].data.clone())
        else :
            self.weight = Parameter(linear.weight.data.clone())
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))

        self.register_buffer('bias_integer', torch.zeros_like(linear.bias))
        if model_dict is not None :
            try : 
                self.bias = Parameter(model_dict[dict_idx + '.bias'].data.clone())
            except AttributeError:
                self.bias = None
        else :
            try:
                self.bias = Parameter(linear.bias.data.clone())
            except AttributeError:
                self.bias = None

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False

    def forward(self, x, prev_act_scaling_factor=None):
        """
        using quantized weights to forward activation x
        """
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]

        if not self.full_precision_flag:
            if self.quant_mode == "symmetric":
                self.weight_function = SymmetricQuantFunction.apply
            elif self.quant_mode == "asymmetric":
                self.weight_function = AsymmetricQuantFunction.apply
            else:
                raise ValueError("unknown quant mode: {}".format(self.quant_mode))


            w = self.weight
            w_transform = w.data.detach()
            # calculate the quantization range of weights and bias
            if self.per_channel:
                w_min, _ = torch.min(w_transform, dim=1, out=None)
                w_max, _ = torch.max(w_transform, dim=1, out=None)
                if self.quantize_bias:
                    b_min = self.bias.data
                    b_max = self.bias.data
            else:
                w_min = w_transform.min().expand(1)
                w_max = w_transform.max().expand(1)
                if self.quantize_bias:
                    b_min = self.bias.data.min()
                    b_max = self.bias.data.max()

            # perform the quantization
            if self.quant_mode == 'symmetric':
                self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max,
                                                                                self.per_channel)
                self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.fc_scaling_factor)

                bias_scaling_factor = self.fc_scaling_factor.view(1, -1) * prev_act_scaling_factor.view(1, -1)
                self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor)
            else:
                raise Exception('For weight, we only support symmetric quantization.')

            prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
            x_int = x / prev_act_scaling_factor
            correct_output_scale = bias_scaling_factor[0].view(1, -1)

            if not self.is_classifier:
                return (ste_round.apply(F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)) * correct_output_scale, self.fc_scaling_factor)
                # return (F.linear(x_int, self.weight_integer, self.bias_integer) * correct_output_scale, self.fc_scaling_factor)
            else:
                return ste_round.apply(F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)) * correct_output_scale
        else:
            return (F.linear(x, weight=self.weight, bias=self.bias), None)


class QuantAct(Module):
    """
    Class to quantize given activations
    Parameters:
    ----------
    activation_bit : int, default 4
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    act_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    fixed_point_quantization : bool, default False
        Whether to skip deployment-oriented operations and use fixed-point rather than integer-only quantization.
    """

    def __init__(self,
                 activation_bit=4,
                 act_range_momentum=0.95,
                 full_precision_flag=False,
                 running_stat=True,
                 quant_mode="symmetric",
                 fix_flag=False,
                 act_percentile=0,
                 fixed_point_quantization=False):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.fix_flag = fix_flag
        self.act_percentile = act_percentile
        self.fixed_point_quantization = fixed_point_quantization

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('act_scaling_factor', torch.zeros(1))

        self.register_buffer('pre_weight_scaling_factor', torch.ones(1))
        self.register_buffer('identity_weight_scaling_factor', torch.ones(1))
        self.register_buffer('concat_weight_scaling_factor', torch.ones(1))
        # self.register_buffer('isDaq', torch.zeros(1, dtype=torch.bool))

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, " \
               "quant_mode={3})".format(self.__class__.__name__, self.activation_bit,
                                       self.full_precision_flag, self.quant_mode)

    def fix(self):
        """
        fix the activation range by setting running stat to False
        """
        self.running_stat = False
        self.fix_flag = True

    def unfix(self):
        """
        unfix the activation range by setting running stat to True
        """
        self.running_stat = True
        self.fix_flag = False

    def forward(self, x, pre_act_scaling_factor=None, pre_weight_scaling_factor=None, identity=None,
                identity_scaling_factor=None, identity_weight_scaling_factor=None, concat=None,
                concat_scaling_factor=None, concat_weight_scaling_factor=None):
        """
        x: the activation that we need to quantize
        pre_act_scaling_factor: the scaling factor of the previous activation quantization layer
        pre_weight_scaling_factor: the scaling factor of the previous weight quantization layer
        identity: if True, we need to consider the identity branch
        identity_scaling_factor: the scaling factor of the previous activation quantization of identity
        identity_weight_scaling_factor: the scaling factor of the weight quantization layer in the identity branch
        Note that there are two cases for identity branch:
        (1) identity branch directly connect to the input featuremap
        (2) identity branch contains convolutional layers that operate on the input featuremap
        """
        if type(x) is list:
            x = x[0]

        # perform the quantization
        if not self.full_precision_flag:
            if self.quant_mode == "symmetric":
                self.act_function = SymmetricQuantFunction.apply
            elif self.quant_mode == "asymmetric":
                self.act_function = AsymmetricQuantFunction.apply
            else:
                raise ValueError("unknown quant mode: {}".format(self.quant_mode))

            # calculate the quantization range of the activations
            if self.running_stat:
                if self.act_percentile == 0:
                    x_min = x.data.min()
                    x_max = x.data.max()
                elif self.quant_mode == 'symmetric':
                    x_min, x_max = get_percentile_min_max(x.detach().view(-1), 100 - self.act_percentile,
                                                    self.act_percentile, output_tensor=True)
                # Note that our asymmetric quantization is implemented using scaled unsigned integers without zero_points,
                # that is to say our asymmetric quantization should always be after ReLU, which makes
                # the minimum value to be always 0. As a result, if we use percentile mode for asymmetric quantization,
                # the lower_percentile will be set to 0 in order to make sure the final x_min is 0.
                elif self.quant_mode == 'asymmetric':
                    x_min, x_max = get_percentile_min_max(x.detach().view(-1), 0, self.act_percentile, output_tensor=True)

                # Initialization
                if self.x_min == self.x_max:
                    self.x_min += x_min
                    self.x_max += x_max

                # use momentum to update the quantization range
                elif self.act_range_momentum == -1:
                    self.x_min = min(self.x_min, x_min)
                    self.x_max = max(self.x_max, x_max)
                else:
                    self.x_min = self.x_min * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
                    self.x_max = self.x_max * self.act_range_momentum + x_max * (1 - self.act_range_momentum)

            if self.quant_mode == 'symmetric':
                self.act_scaling_factor = symmetric_linear_quantization_params(self.activation_bit,
                                                                           self.x_min, self.x_max, False)
            # Note that our asymmetric quantization is implemented using scaled unsigned integers
            # without zero_point shift. As a result, asymmetric quantization should be after ReLU,
            # and the self.act_zero_point should be 0.
            else:
                self.act_scaling_factor, self.act_zero_point = asymmetric_linear_quantization_params(
                    self.activation_bit, self.x_min, self.x_max, True)
            if (pre_act_scaling_factor is None) or (self.fixed_point_quantization == True):
                # this is for the case of input quantization,
                # or the case using fixed-point rather than integer-only quantization
                quant_act_int = self.act_function(x, self.activation_bit, self.act_scaling_factor)
            elif type(pre_act_scaling_factor) is list:
                # this is for the case of multi-branch quantization
                branch_num = len(pre_act_scaling_factor)
                quant_act_int = x
                start_channel_index = 0
                for i in range(branch_num):
                    quant_act_int[:, start_channel_index: start_channel_index + channel_num[i], :, :] \
                        = fixedpoint_fn.apply(x[:, start_channel_index: start_channel_index + channel_num[i], :, :],
                                              self.activation_bit, self.quant_mode, self.act_scaling_factor, 0,
                                              pre_act_scaling_factor[i],
                                              pre_act_scaling_factor[i] / pre_act_scaling_factor[i])
                    start_channel_index += channel_num[i]
            else:
                if identity is None and concat is None:
                    if pre_weight_scaling_factor is None:
                        pre_weight_scaling_factor = self.pre_weight_scaling_factor
                    quant_act_int = fixedpoint_fn.apply(x, self.activation_bit, self.quant_mode,
                                                        self.act_scaling_factor, 0, pre_act_scaling_factor,
                                                        pre_weight_scaling_factor)
                elif identity is not None:
                    if identity_weight_scaling_factor is None:
                        identity_weight_scaling_factor = self.identity_weight_scaling_factor
                    quant_act_int = fixedpoint_fn.apply(x, self.activation_bit, self.quant_mode,
                                                        self.act_scaling_factor, 1, pre_act_scaling_factor,
                                                        pre_weight_scaling_factor,
                                                        identity, identity_scaling_factor,
                                                        identity_weight_scaling_factor)
                else:
                    if concat_weight_scaling_factor is None:
                        concat_weight_scaling_factor = self.concat_weight_scaling_factor
                    quant_act_int = fixedpoint_fn.apply(x, self.activation_bit, self.quant_mode,
                                                         self.act_scaling_factor, 2, pre_act_scaling_factor,
                                                         pre_weight_scaling_factor,
                                                         concat, concat_scaling_factor,
                                                         concat_weight_scaling_factor)
            return [quant_act_int], [self.act_scaling_factor.view(-1, 1, 1)]
        else:
            return x, None
