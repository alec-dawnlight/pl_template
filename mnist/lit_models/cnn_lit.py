import argparse
import itertools
import torch

from .base import BaseLitModel
from .metrics import PrecisionRecallCurve
from .util import to_onehot
import logging
logger = logging.getLogger('lightning')


class CNNLitModel(BaseLitModel):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args)

        self.inverse_mapping = {val: ind for ind, val in enumerate(self.model.data_config["mapping"])}
        # self.loss_fn = torch.nn.KLDivLoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.val_pr_curve = PrecisionRecallCurve(len(self.inverse_mapping))
        self.test_pr_curve = PrecisionRecallCurve(len(self.inverse_mapping))

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=1e-3)
        return parser

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)

        loss = self.loss_fn(logprobs, y)
        self.log("train_loss", loss)

        probs = torch.softmax(logits, dim=1)
        self.train_acc(probs, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        
        loss = self.loss_fn(logprobs, y)
        self.log("val_loss", loss, prog_bar=True)
    
        probs = torch.softmax(logits, dim=1)
        self.val_acc(probs, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_pr_curve(probs, y)
        # self.log("val_pr_curve", self.val_pr_curve, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        self.test_acc(probs, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_pr_curve(probs, y)
        # self.log("test_prcurve", self.test_pr_curve, on_step=False, on_epoch=True, prog_bar=True)
