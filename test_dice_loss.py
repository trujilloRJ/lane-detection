import torch
from network import dice_loss

pred = torch.tensor([
    [[50., -50.], 
     [50., -50.]],
    [[50., 50.], 
     [50., -50.]]
], dtype=torch.float32, requires_grad=False)

target = torch.tensor([
    [[1, 1], 
     [1, 0]],
    [[1., 1.], 
     [1, 1]]
], dtype=torch.float32, requires_grad=False)

print(dice_loss(pred, target))