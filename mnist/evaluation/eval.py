"""Run validation test for mnist."""

import os
import argparse
import time
import unittest
import torch
import pytorch_lightning as pl
from mnist.data import MNIST
from mnist.run_pytorch import ModelRunner

_TEST_ACCURACY = 0.98

class TestEvaluateModel(unittest.TestCase):
    """Evaluate ParagraphTextRecognizer on the IAMParagraphs test dataset."""

    @torch.no_grad()
    def test_evaluate(self):
        dataset = MNIST(argparse.Namespace(batch_size=128, num_workers=10))
        dataset.prepare_data()
        dataset.setup()

        model_runner = ModelRunner()
        trainer = pl.Trainer(gpus=1)

        start_time = time.time()
        metrics = trainer.test(model_runner.lit_model, datamodule=dataset)
        end_time = time.time()

        test_acc = round(metrics[0]["test_acc"], 2)
        time_taken = round((end_time - start_time) / 60, 2)

        print(f"Accuracy: {test_acc}, time_taken: {time_taken} m")
        self.assertEqual(test_acc, _TEST_ACCURACY)
        self.assertLess(time_taken, 45)

