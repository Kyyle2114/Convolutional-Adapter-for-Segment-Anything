import cv2
import numpy as np

def make_box_prompt(mask: np.array, 
                    scale_factor: float = 1.0,
                    return_xyxy: bool = True) -> np.array:
    """
    Generate a box prompt that includes the mask(True region).

    Args:
        mask (np.array): given mask 
        scale_factor (float, optional): Adjust the size of the resulting box. 
                                        It is the same as cv2.boundingRect when set to 1.0. Defaults to 1.0.
        return_xyxy(bool, optional) : if True, return the coordinates in the form of (x1, y1, x2, y2).
                                      if Fale, return the coordinates in the form of (x, y, w, h). Defaults to True.
    Returns:
        np.array: coords of box, (x1, y1, x2, y2)
    """
    
    # mask shape 
    H, W = mask.shape[-2:]
    
    non_zero_points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(non_zero_points)
    
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    x = x - int((new_w - w) / 2)
    y = y - int((new_h - h) / 2)
    
    # if the new coords are outside the box 
    x = x if x > 0 else 0 
    y = y if y > 0 else 0 
    
    if x + new_w >= W:
        new_w = W - x - 1
    
    if y + new_h >= H:
        new_h = H - y - 1
    
    if return_xyxy:
        x1, y1, x2, y2 = x, y, x + new_w, y + new_h
        rect = np.array((x1, y1, x2, y2))
        
    else:
        rect = np.array((x, y, new_w, new_h))

    return rect