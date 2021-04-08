from torch.utils.tensorboard import SummaryWriter
from os.path import join
from pprint import pprint


class Monitor():
    def __init__(self, folder=None):
        """logs parameters into tensorboard

        :param folder: folder name, defaults to None
        :type folder: str, optional
        """        
        self.folder = folder
        self.writer = SummaryWriter(self.folder)
        self.global_step = 0

    def step_train(self, train_dict):
        """Writes params for one step of the training.

        :param train_dict: dict of metrics to be stored
        :type train_dict: dict
        """        
        for k, v in train_dict.items():
            self.writer.add_scalar(
                k+'/train', v, global_step=self.global_step)
        self.global_step += 1

    def step_val(self, val_dict):
        """Writes params for one step of the validation.

        :param val_dict: dict of metrics to be stored
        :type val_dict: dict
        """        
        print('\n #### Validation')
        pprint(val_dict)
        for k, v in val_dict.items():
            self.writer.add_scalar(
                k+'/val', v, global_step=self.global_step)
