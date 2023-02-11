import torch
import numpy as np
from multiprocessing import Manager, Process, freeze_support
import torch.distributed as dist
from .coco import COCODataset
import random
import time


# freeze_support()


def single_worker(dataset, num_workers, rank, world_size, num_id, data):
    import time

    cache_size = 10
    idx = 0

    data["run"] = True
    data["order_map"] = []
    data["batch_size"] = 0

    data[str(num_id)] = []

    while not (len(data) == num_workers + 3 and data["batch_size"] > 0):
        pass

    while data["run"]:

        while len(data[str(num_id)]) >= cache_size:
            pass

        t0 = time.time()
        local_data = []
        order_map = data["order_map"]
        batch_size = data["batch_size"]

        start_idx = (idx * world_size + rank) * batch_size
        # print(batch_size, start_idx)
        for j in range(start_idx, start_idx + batch_size):
            if j % num_workers == num_id:
                img, targets, _, _, segms = dataset[order_map[j]]
                local_data.append([img, targets, segms])

        data["%d_%d" % (num_id, idx)] = local_data
        idx += 1

        # print(num_id, len(local_data), time.time()-t0)



def get_data_source(dataset: COCODataset, rank, world_size=1, num_workers=4):
    train_data = Manager().dict()
    processes = [Process(target=single_worker,
                         args=(dataset,
                               num_workers,
                               rank,
                               world_size,
                               num_id,
                               train_data))
                 for num_id in range(num_workers)]
    return train_data, processes


class Fetcher:

    def __init__(self, data, batch_len, data_len, batch_size, num_workers, world_size):

        self.__len = batch_len
        self.__data_len = data_len
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.world_size = world_size

    def __init_order_map(self, rand=True):
        order_map = [i for i in range(self.__data_len)]
        if rand:
            random.shuffle(order_map)
        return order_map

    def __len__(self):
        return self.__len

    def get_fetcher(self, rand=True):
        return self.__fetcher(rand)

    def __fetcher(self, rand=True):

        order_map = self.__init_order_map(rand)

        for i in range(self.__len):
            t0 = time.time()

            self.data["order_map"] = order_map
            self.data["batch_size"] = self.batch_size

            t1 = time.time()

            while not all([("%d_%d" % (num_id, i) in self.data) for num_id in range(self.num_workers)]):
                pass

            t2 = time.time()

            result = self.__sum_up(i)

            t3 = time.time()
            # print("def time:", t1 - t0, "s")
            # print("wait time:", t2 - t1, "s")
            # print("sumup time:", t3 - t2, "s")
            yield result

    def __sum_up(self, idx):
        self.__temp_data = []
        t0 = time.time()
        for num_id in range(self.num_workers):
            # print(train_data)
            self.__temp_data += self.data["%d_%d" % (num_id, idx)]
            del self.data["%d_%d" % (num_id, idx)]
        # print("get data time:", time.time() - t0, "s")

        imgs = []
        targetss =[]
        segments = []

        [[imgs.append(img), targetss.append(targets), segments.append(segms)]
          for img, targets, segms in self.__temp_data]


        if self.world_size > 1:
            dist.barrier()
        imgs = torch.from_numpy(np.array(imgs))
        targetss = torch.from_numpy(np.array(targetss))

        return imgs, targetss, segments


class FastMaskDataloader:

    def __init__(self, dataset, batch_size, rank=0, world_size=1, num_workers=4):
        data, self.processes = get_data_source(dataset, rank, world_size, num_workers)
        self.batch_size = batch_size // world_size
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers

        self.__data_len = len(dataset)
        self.__len = self.__data_len // batch_size
        self.fetcher = Fetcher(data, self.__len, self.__data_len, self.batch_size, num_workers, world_size)

    def __len__(self):
        return self.__len

    def add_progress(self, **kwargs):
        self.processes.append(Process(**kwargs))

    def run(self):
        [progress.start() for progress in self.processes]
        [progress.join() for progress in self.processes]
