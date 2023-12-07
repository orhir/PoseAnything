from mmcv.runner import Hook
from mmpose.utils import get_root_logger
from torch.utils.data import DataLoader


class ShufflePairedSamplesHook(Hook):
    """Non-Distributed ShufflePairedSamples.
    After each training epoch, run FewShotKeypointDataset.random_paired_samples()
    """

    def __init__(self,
                 dataloader,
                 interval=1):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f'dataloader must be a pytorch DataLoader, '
                            f'but got {type(dataloader)}')

        self.dataloader = dataloader
        self.interval = interval
        self.logger = get_root_logger()

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        if not self.every_n_epochs(runner, self.interval):
            return
        # self.logger.info("Run random_paired_samples()")
        # self.logger.info(f"Before: {self.dataloader.dataset.paired_samples[0]}")
        self.dataloader.dataset.random_paired_samples()
        # self.logger.info(f"After: {self.dataloader.dataset.paired_samples[0]}")
