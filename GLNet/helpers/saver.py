import os
import torch
from collections import OrderedDict

class Saver(object):
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        self.task_name = args.task_name

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def save_checkpoint(self, state):
        """
        Save checkpoint to disk
        """
        filename = os.path.join(self.model_path, f'{self.task_name}.pth')
        torch.save(state, filename)

    def save_experiment_config(self):
        pass
