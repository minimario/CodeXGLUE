# import parser
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

# create a neural net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def init_ddp(args):
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   

    args = parser.parse_args() 
    init_ddp(args)
    model = Net()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # create some data
    n_examples = 1000
    x, y = torch.randn(n_examples, 10), torch.randn(n_examples, 1)
    x = x.cuda()
    y = y.cuda()

    # make the dataloader
    dataset = torch.utils.data.TensorDataset(x, y)
    sampler = DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, sampler=sampler)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # train the model
    for epoch in range(3):
        print(epoch)
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

    if args.local_rank == 0:
        print('Finished Training')
        model.eval()
        # do eval
        eval_x = torch.randn(100, 10)
        eval_x = eval_x.cuda()
        with torch.no_grad():
            eval_y = model(eval_x)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
    
    print("done!")


if __name__ == '__main__':
    main()

