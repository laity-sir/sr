import h5py
from torchvision import transforms
import sys
import queue
import threading
import torch
import config
import numpy as np
from torch.utils.data import Dataset, DataLoader
sys.path.append('./data')

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file=h5_file
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            idx=idx%len(f['hr'])
            lr=transforms.ToTensor()(f['lr'][idx])
            hr=transforms.ToTensor()(f['hr'][idx])
            return {"lr":lr,"hr":hr}
    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr'])

class EvalDataset(Dataset):
    def __init__(self, h5_file,mode='y'):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.mode=mode
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr=f['lr'][str(idx)][:,:]
            hr=f['hr'][str(idx)][:, :]
            if lr.ndim==2:
                lr=np.expand_dims(lr,2)
                hr=np.expand_dims(hr,2)
            lr=transforms.ToTensor()(lr)
            hr=transforms.ToTensor()(hr)
            return lr,hr
    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    num=40
    col=8
    row=int(num/8)
    dataset=TrainDataset('../train/div2k_{}.h5'.format(config.scale))
    # print(dataset[0][1])
    index = np.random.randint(1, len(dataset), num)
    print(index)
    hh=[dataset[i]['lr'] for i in index]
    ###可视化图片
    for i in range(num):
        for j in range(8):
            # plt.figure()
            plt.subplot(row, col, i+1)
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(transforms.ToPILImage()(hh[i]), cmap='gray')
    plt.show()

