from typing import Any, Dict
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        input_dim = np.prod(data_config["input_dims"])
        num_classes = len(data_config["mapping"])

        fc1_dim = self.args.get("fc1", 1024)
        fc2_dim = self.args.get("fc2", 128)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--fc1", type=int, default=1024)
        parser.add_argument("--fc2", type=int, default=128)
        return parser

