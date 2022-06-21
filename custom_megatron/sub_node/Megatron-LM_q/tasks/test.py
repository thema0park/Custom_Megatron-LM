import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def example(rank, world_size):
    print(rank)
    # create default process group
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("gloo", init_method='tcp://166.104.112.80:23456',
                        rank=1, world_size=6)
    print("conncet done!!")
    # create local model
    model = nn.Linear(10, 10).to(1)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[1])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(0))
    labels = torch.randn(20, 10).to(0)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    example(1,2)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    #os.environ["MASTER_ADDR"] = "166.104.112.80"
    #os.environ["MASTER_PORT"] = "29500"
    main()
