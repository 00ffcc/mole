import math
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    带有线性预热 (Warmup) 和 Cosine 退火衰减的学习率调度器。
    学习率先从一个较小的值（通常是0）线性增加到基础学习率 (base_lr)，
    然后按照 Cosine 曲线从 base_lr 衰减到最小学习率 (min_lr)。

    参数:
        optimizer (Optimizer): 被包装的优化器。
        num_warmup_steps (int): 线性预热的步数。
        num_training_steps (int): 总的训练步数 (预热 + 衰减)。
        min_lr_ratio (float, optional): 最小学习率与基础学习率的比率。
                                      最终学习率 = base_lr * min_lr_ratio。
                                      默认为 0.0。
        last_epoch (int, optional): 上一个 epoch 的索引。用于恢复训练。
                                    默认为 -1。

    注意:
        这个调度器是基于 **步数 (step)** 而不是 epoch 数进行更新的。
        请确保在每个训练步 (optimizer.step() 之后) 调用 scheduler.step()。
    """
    def __init__(self,
                 optimizer: Optimizer,
                 num_warmup_steps: int,
                 num_training_steps: int,
                 min_lr_ratio: float = 0.0,
                 last_epoch: int = -1):

        if num_warmup_steps < 0:
            raise ValueError("num_warmup_steps must be non-negative.")
        if num_training_steps < num_warmup_steps:
            raise ValueError("num_training_steps must be greater than or equal to num_warmup_steps.")
        if not 0.0 <= min_lr_ratio <= 1.0:
             raise ValueError("min_lr_ratio must be between 0.0 and 1.0.")

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio

        # 基类 __init__ 应该在最后调用，因为它会调用 get_lr() (如果 last_epoch != -1)
        # 确保所有需要的成员变量 (self.num_warmup_steps 等) 已经设置
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        根据当前步数 (self.last_epoch) 计算学习率缩放因子。
        返回一个包含每个参数组的学习率的列表。
        """
        # self.last_epoch 是当前所在的步数 (0-indexed)
        current_step = self.last_epoch

        # 1. Warmup 阶段
        if current_step < self.num_warmup_steps:
            # 线性增加: 从 0 到 1
            scale_factor = float(current_step) / float(max(1.0, self.num_warmup_steps))
        # 2. Cosine Decay 阶段
        elif current_step < self.num_training_steps:
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            # Cosine 退火: 从 1 到 0
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # 缩放到 [min_lr_ratio, 1.0] 范围
            scale_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
        # 3. 训练结束后阶段
        else:
            # 保持在最小学习率
            scale_factor = self.min_lr_ratio

        # 应用缩放因子到每个参数组的基础学习率 (base_lrs 由父类初始化时获取)
        return [base_lr * scale_factor for base_lr in self.base_lrs]