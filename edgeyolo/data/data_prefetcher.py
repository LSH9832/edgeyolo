import torch
from .samplers import InfiniteSampler, YoloBatchSampler
from .dataloading import worker_init_reset_seed, DataLoader
# import cv2
#
# cv2.setNumThreads(params.num_threads)


class MaskDataLoader:

    def __init__(
            self,
            dataset,
            batch_size: int,
            rank=0,
            world_size=1,
            num_workers=4,
            train_mask=True
    ):
        self.dataset = dataset
        self.train_mask = train_mask
        sampler = InfiniteSampler(
            len(self.dataset),
            seed=0,
            rank=rank,
            world_size=world_size
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,             # DROP LAST BATCH IF ITS BATCH SIZE IS SMALLER
            mosaic=True,
        )

        dataloader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed
        }
        self.train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        self.prefetcher = DataPrefetcher(self.train_loader, train_mask)

    def next(self):
        return self.prefetcher.next()

    def close_mosaic(self):
        self.train_loader.close_mosaic()

    def __len__(self):
        return len(self.train_loader)


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader, train_mask=True):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image

        self.next_input = None
        self.next_target = None
        self.next_segms = None
        self.train_mask = train_mask

        self.preload()

    def preload(self):
        try:
            if self.train_mask:
                self.next_input, self.next_target, _, _, self.next_segms = next(self.loader)
            else:
                self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            if self.train_mask:
                self.next_segms = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)
            if self.train_mask:
                self.next_segms = self.next_segms.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if self.train_mask:
            segms = self.next_segms
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if self.train_mask:
            if segms is not None:
                segms.record_stream(torch.cuda.current_stream())
        self.preload()
        if self.train_mask:
            return input, target, segms
        else:
            return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
