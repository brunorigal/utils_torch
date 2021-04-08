import torch
from os.path import join
import os


class Saver():
    def __init__(self, model, module_dict={}, folder=None, save_min=False, save_freq=10,
        start_epoch=0):
        """Saves a torch module and a dict of modules

        :param model: model
        :type model: torch.nn.Module
        :param module_dict: dict of modules, defaults to {}
        :type module_dict: dict, optional
        :param folder: forlder name, defaults to None
        :type folder: str, optional
        :param save_min: if true the best value of the target metric is the minimum value, defaults to False
        :type save_min: bool, optional
        :param save_freq: frequency of saving, defaults to 10
        :type save_freq: int, optional
        :param start_epoch: first epoch, defaults to 0
        :type start_epoch: int, optional
        """        
        self.model = model
        self.module_dict = module_dict
        self.folder = folder
        self.save_min = save_min
        self.save_freq = save_freq
        self.best_crit = 10000 if save_min else -10000
        self.epoch = start_epoch
        self.file_saved = None

    def save(self, crit_val):
        """Saves a new model if crit_val is better than the best stored best criterion.

        :param crit_val: target metric used for storing or not the model 
        :type crit_val: float
        """        
        if (self.save_min and crit_val < self.best_crit) or ((not self.save_min) and crit_val > self.best_crit):
            file_saved = join(
                self.folder, f'best_epoch_{self.epoch}_crit_{crit_val:.3f}.pth')
            torch.save({'model': self.model.state_dict(),
                        'epoch': self.epoch,
                        **{k: mod.state_dict() for k, mod in self.module_dict.items()}
                        }, file_saved)
            if self.file_saved is not None:
                os.remove(self.file_saved)
            self.file_saved = file_saved
            self.best_crit = crit_val
        if ((self.epoch+1) % self.save_freq) == 0:
            torch.save({'model': self.model.state_dict(),
                        'epoch': self.epoch,
                        **{k: mod.state_dict() for k, mod in self.module_dict.items()}
                        }, join(self.folder, f'epoch_{self.epoch}_crit_{crit_val:.3f}.pth'))
        self.epoch += 1
