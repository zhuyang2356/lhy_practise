import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
        optimizer:Optimizer,
        num_warmup_steps:int,
        num_training_steps:int,
        num_cycles:float=0.5,
        last_epoch:int=-1,):
    """

    :param optimizer:
    :param num_warmup_steps:
    :param num_training_steps:
    :param num_cycles:
    :param last_epoch:
    :return:
    """
    # TODO
    # 理解warmup的原理
    def lr_lambda(current_step):
        # WarmUp
        if current_step<num_warmup_steps:
            return float(current_step)/float(max(1, num_warmup_steps))
        # decadence
        progress=float(current_step - num_warmup_steps) / \
                 float(max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    return LambdaLR(optimizer,lr_lambda,last_epoch)