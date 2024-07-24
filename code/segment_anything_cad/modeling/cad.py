import torch
import torch.nn as nn

from typing import Tuple

class ConvAdapter(nn.Module):
    def __init__(self,
                 in_chans: int = 6,
                 out_chans: int = 256):
        """
        Convolutional Adapter for SAM
        The output of the ConvAdapter is added to the image embedding from ViT(image encoder).

        Args:
            in_chans (int, optional): input channels. Defaults to 6.
            out_chans (int, optional): output channels. Defaults to 256.
        """
        
        super(ConvAdapter, self).__init__()
        
        self.conv_up = nn.Conv2d(
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=1,
            stride=1,
            padding='same'
        )
        
        self.conv_block1 = ConvBlock(channel=out_chans)
        self.conv_block2 = ConvBlock(channel=out_chans)
        self.conv_block3 = ConvBlock(channel=out_chans)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((64, 64))
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # N: batch size  
        # x: (N, C, H, W) size tensor
        m = torch.amax(x, dim=(2, 3), keepdim=True) + 1e-5
        x = x / m
        
        x = self.conv_up(x)
        
        # H, W: 512 -> 256
        p1 = self.conv_block1(x)
        
        # 256 -> 128
        p2 = self.conv_block2(p1)
        
        # 128 -> 64
        p3 = self.conv_block3(p2)
        
        p1 = self.avg_pool(p1)
        p2 = self.avg_pool(p2)
        p3 = self.avg_pool(p3)
        
        x = p1 + p2 + p3
        
        return torch.tanh(x)

class ConvBlock(nn.Module):
    def __init__(self,
                 channel: int = 256):
        
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=channel, 
            out_channels=channel, 
            kernel_size=3, 
            stride=1,
            dilation=2,
            padding='same'
        )
        
        self.bn = nn.BatchNorm2d(channel)
                
        self.act_layer = nn.LeakyReLU(0.1)
        
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x 
        
        x = self.act_layer(self.bn(self.conv(x)))
                
        # skip connection
        x = x + shortcut
        
        x = self.avg_pool(x)
        
        return x 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)