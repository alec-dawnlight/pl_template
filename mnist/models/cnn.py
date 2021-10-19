import argparse
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    '''
    3x3 conv w/ padding = 1 and relu
    '''
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class CNN(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace=None) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        self.img_size = 28

        input_dims = data_config.get("input_dims")
        num_classes = len(data_config.get("mapping"))

        conv_dim = self.args.get("conv_dim", 64)
        fc_dim = self.args.get("fc_dim", 128)

        self.c1 = ConvBlock(input_dims[0], conv_dim)
        self.c2 = ConvBlock(conv_dim, conv_dim)
        self.dropout = nn.Dropout(0.25)
        self.max_pool = nn.MaxPool2d(2)

        conv_output_size = IMAGE_SIZE//2
        fc_input_dim = int(conv_dim * conv_output_size**2)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE
        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == self.img_size
        x = self.c1(x)
        x = self.c2(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--conv_dim', type=int, default=64)
        parser.add_argument('--fc_dim', type=int, default=128)
        return parser

