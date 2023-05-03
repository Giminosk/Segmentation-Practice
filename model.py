import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    


class UNetLike(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoders = nn.ModuleList(
            [
                ConvBlock(in_channels, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                ConvBlock(256, 512)
            ]
        )

        self.bottleneck = ConvBlock(512, 1024)

        self.transpose_convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                ConvBlock(1024, 512),
                ConvBlock(512, 256),
                ConvBlock(256, 128),
                ConvBlock(128, 64)
            ]
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

        self.process_blocks = nn.ModuleList(
            [
                ConvBlock(512, 512),
                ConvBlock(256, 256),
                ConvBlock(128, 128),
                ConvBlock(64, 64)
            ]
        )

    
    def forward(self, x):
        skip_connections = []

        for down in self.encoders:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.decoders)):
            x = self.transpose_convs[idx](x)
            skip = self.process_blocks[idx](skip_connections[idx])
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx](x)

        x = self.final_conv(x)

        return x
    
