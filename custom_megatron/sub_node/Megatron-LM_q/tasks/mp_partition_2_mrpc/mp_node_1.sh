#!/bin/bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=put your ethernet name
export CUDA_VISIBLE_DEVICES=0

GPUS_PER_NODE=1
MASTER_ADDR=MASTER_NODE_IP
MASTER_PORT=MASTER_NODE_PORT
NNODES=2 #NODE_NUM
NODE_RANK=1 #NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
TRAIN_DATA="put training data path"
VALID_DATA="put validation data path"
PRETRAINED_CHECKPOINT="put pretrained BERT-large with Megatron .pt file path"
VOCAB_FILE="put Google BERT vocab.txt file path"
CHECKPOINT_PATH="put save path"
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch $DISTRIBUTED_ARGS ../main.py \
               --task put your task_name \
               --quantize_forward \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceLowerCase \
               --vocab-file $VOCAB_FILE \
               --epochs 4 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
	           --pipeline-model-parallel-size 2 \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --global-batch-size 32 \
               --micro-batch-size 32 \
               --lr 2.0e-5 \
               --lr-decay-style linear \
               --lr-warmup-fraction 0.1 \
               --seq-length 128 \
               --max-position-embeddings 128 \
               --log-interval 100 \
               --eval-interval 100 \
               --eval-iters 10 \
               --weight-decay 1.0e-2 \
	           --local_rank 0 \
	           --finetune \
	           --DDP-impl local
