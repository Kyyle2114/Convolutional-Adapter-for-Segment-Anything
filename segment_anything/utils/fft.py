"""
Code Reference : https://github.com/tianrun-chen/SAM-Adapter-PyTorch/blob/main/models/mmseg/models/sam/image_encoder.py
"""

import torch

def extract_freq_components(x: torch.Tensor, 
                            tau_rate: float = 0.25) -> torch.Tensor:
    """
    Extract High / Low frequency components using FFT / IFFT

    Args:
        x (torch.Tensor): inpurt tensor, (N, C, H, W). N: batch size 
        tau_rate (float, optional): mask ratio. Defaults to 0.25.

    Returns:
        torch.Tensor: high frequency components.
    """
    mask = torch.zeros(x.shape).to(x.device)
    W, H = x.shape[-2:]
    
    # W, H = 1024, tau_rate = 0.25 -> masking_area = 256
    masking_area = int((W * H * tau_rate) ** .5 // 2)
    mask[:, :, W // 2 - masking_area : W // 2 + masking_area, H // 2 - masking_area : H // 2 + masking_area] = 1
    
    fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
    
    # High frequency components
    high_fft = fft * (1 - mask)
    high_fr, high_fi = high_fft.real, high_fft.imag
    
    high_fft_hires = torch.fft.ifftshift(torch.complex(high_fr, high_fi))
    high_inv = torch.fft.ifft2(high_fft_hires, norm="forward").real
    high_inv = torch.abs(high_inv)

    return high_inv