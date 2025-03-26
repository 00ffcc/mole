# from https://github.com/pytorch/pytorch/blob/master/torch/optim/adamw.py
from typing import cast, List, Optional, Tuple, Union

import torch
from torch import Tensor

def adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    beta1: Union[Tensor, float],
    beta2: Union[Tensor, float],
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
):

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step_t
        bias_correction2 = 1 - beta2**step_t

        step_size = lr / bias_correction1
        step_size_neg = step_size.neg()

        bias_correction2_sqrt = bias_correction2.sqrt()

        denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
        param.addcdiv_(exp_avg, denom)


