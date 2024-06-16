"""
Code Reference : SAM-Adapter, https://github.com/tianrun-chen/SAM-Adapter-PyTorch/blob/main/models/iou_loss.py
"""

import torch

class IoULoss(torch.nn.Module):
    def __init__(self):
        """
        IoU Loss for Pytorch 
        Input tensor should be 3-dimensional(N(batch_size), H, W), contains values of either 0(False) or 1(True).
        """
        super(IoULoss, self).__init__()

    def _iou(self, pred, target):
        smooth = 1e-3
        
        inter = (pred * target).sum(dim=(1, 2))
        union = (pred + target).sum(dim=(1, 2)) - inter
        iou = (inter + smooth) / (union + smooth)
        
        iou = 1 - iou

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)