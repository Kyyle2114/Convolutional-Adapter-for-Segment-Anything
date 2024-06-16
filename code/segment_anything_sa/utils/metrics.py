import torch

def Dice(pred: torch.Tensor, 
         target: torch.Tensor) -> torch.Tensor:
    """
    Dice metric for segmentation.

    Args:
        pred (torch.Tensor): (N(batch_size), H, W) size tensor 
        target (torch.Tensor): (N(batch_size), H, W) size tensor

    Returns:
        torch.Tensor: average Dice score
    """
    smooth = 1e-3
    intersection = torch.sum(pred * target, dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2)) + smooth)
    
    return dice.mean()

def IoU(pred: torch.Tensor, 
        target: torch.Tensor) -> torch.Tensor:
    """
    IoU metric for segmentation.

    Args:
        pred (torch.Tensor): (N(batch_size), H, W) size tensor 
        target (torch.Tensor): (N(batch_size), H, W) size tensor

    Returns:
        torch.Tensor: average IoU score
    """
    smooth = 1e-3
    inter = torch.sum(pred * target, dim=(1, 2))
    union = torch.sum(pred + target, dim=(1, 2)) - inter 
    iou = (inter + smooth) / (union + smooth)
    
    return iou.mean()

