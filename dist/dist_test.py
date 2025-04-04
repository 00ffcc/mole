from accelerate import Accelerator
import torch
accelerator = Accelerator()
if accelerator.is_local_main_process:
    m = torch.empty((100, 100), pin_memory=True, device='cpu')