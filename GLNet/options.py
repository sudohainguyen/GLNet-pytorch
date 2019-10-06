###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import argparse
import warnings
from GLNet.utils import PhaseMode

__all__ = ['TrainingOptions']

def _check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """
    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))
    return parsed_args

class TrainingOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description="GLNet for segmenation")
        
        parser.add_argument(
            "--n_class", type=int, default=7, help="segmentation classes"
        )
        parser.add_argument(
            "--data_path", type=str, help="path to dataset where images store"
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default="./checkpoints",
            help="path to store trained model files, no need to include task specific name",
        )
        parser.add_argument(
            "--log_path",
            type=str,
            default="./logs",
            help="path to store tensorboard log files, no need to include task specific name",
        )
        parser.add_argument(
            "--task_name",
            type=str,
            help="task name for naming saved model files and log files",
        )
        parser.add_argument(
            "--mode",
            type=int,
            default=1,
            choices=[1, 2, 3],
            help="mode for training procedure. \
                1: train global branch only. \
                2: train local branch with fixed global branch. \
                3: train global branch with fixed local branch"
        )
        # parser.add_argument(
        #     "--validation", action="store_true", default=False, help="Execute validation after each epoch"
        # )
        parser.add_argument(
            "--validation-steps", type=int, default=1, help="Number of epoch that model will be validated"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="batch size for origin global image (without downsampling)",
        )
        parser.add_argument(
            "--sub_batch_size",
            type=int,
            default=8,
            help="batch size for using local image patches",
        )
        parser.add_argument(
            "--size_g",
            type=int,
            default=508,
            help="size (in pixel) for downsampled global image",
        )
        parser.add_argument(
            "--size_p",
            type=int,
            default=508,
            help="size (in pixel) for cropped local image",
        )
        parser.add_argument(
            "--path_g", type=str, default="", help="name for global model path"
        )
        parser.add_argument(
            "--path_g2l",
            type=str,
            default="",
            help="name for local from global model path",
        )
        parser.add_argument(
            "--path_l2g",
            type=str,
            default="",
            help="name for global from local model path",
        )
        parser.add_argument(
            "--lamb_fmreg",
            type=float,
            default=0.15,
            help="loss weight feature map regularization",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=5,
            help="Number of workers for dataloader"
        )
        parser.add_argument(
            '--backbone',
            type=str,
            default='resnet50',
            help='Backbone model used by FPN.')
        
        self.parser = parser
    
    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr
        if args.mode == 1 or args.mode == 3:
            args.num_epochs = 120
            args.lr = 5e-5
        else:
            args.num_epochs = 50
            args.lr = 2e-5
        
        args.mode = PhaseMode.int_to_phasemode(args.mode)
        return _check_args(args)

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--n_class", type=int, default=7, help="segmentation classes"
        )
        parser.add_argument(
            "--data_path", type=str, help="path to dataset where images store"
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default="./checkpoints",
            help="path to store trained model files, no need to include task specific name",
        )
        parser.add_argument(
            "--log_path",
            type=str,
            default="./logs",
            help="path to store tensorboard log files, no need to include task specific name",
        )
        parser.add_argument(
            "--task_name",
            type=str,
            help="task name for naming saved model files and log files",
        )
        parser.add_argument(
            "--mode",
            type=int,
            default=1,
            choices=[1, 2, 3],
            help="mode for training procedure. \
                1: train global branch only. \
                2: train local branch with fixed global branch. \
                3: train global branch with fixed local branch"
        )
        parser.add_argument(
            "--evaluation", action="store_true", default=False, help="evaluation only"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=6,
            help="batch size for origin global image (without downsampling)",
        )
        parser.add_argument(
            "--sub_batch_size",
            type=int,
            default=6,
            help="batch size for using local image patches",
        )
        parser.add_argument(
            "--size_g",
            type=int,
            default=508,
            help="size (in pixel) for downsampled global image",
        )
        parser.add_argument(
            "--size_p",
            type=int,
            default=508,
            help="size (in pixel) for cropped local image",
        )
        parser.add_argument(
            "--path_g", type=str, default="", help="name for global model path"
        )
        parser.add_argument(
            "--path_g2l",
            type=str,
            default="",
            help="name for local from global model path",
        )
        parser.add_argument(
            "--path_l2g",
            type=str,
            default="",
            help="name for global from local model path",
        )
        parser.add_argument(
            "--lamb_fmreg",
            type=float,
            default=0.15,
            help="loss weight feature map regularization",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=5,
            help="Number of workers for dataloader"
        )

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr
        if args.mode == 1 or args.mode == 3:
            args.num_epochs = 120
            args.lr = 5e-5
        else:
            args.num_epochs = 50
            args.lr = 2e-5
        
        args.mode = PhaseMode.int_to_phasemode(args.mode)
        return args
