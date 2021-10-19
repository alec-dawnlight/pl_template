"""MNIST DataModule"""
import argparse
from typing import Tuple

from PIL import Image
from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms

from mnist.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"

## NOTE: temp fix until https://github.com/pytorch/vision/issues/1938 is resolved
#from six.moves import urllib  # pylint: disable=wrong-import-position, wrong-import-order
#
#opener = urllib.request.build_opener()
#opener.addheaders = [("User-agent", "Mozilla/5.0")]
#urllib.request.install_opener(opener)


class MNIST(BaseDataModule):
    """
    MNIST DataModule.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.dims = (1, 28, 28)  # dims are returned when calling `.size()` on this object.
        self.output_dims = (1,)
        self.mapping = list(range(10))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""
        TorchMNIST(self.data_dir, train=True, download=True)
        TorchMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        mnist_full = TorchMNIST(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_full, [55000, 5000])
        self.data_test = TorchMNIST(self.data_dir, train=False, transform=self.transform)
    
    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "MNIST Dataset\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


    def get_transform(self, image_shape: Tuple[int, int], augment: bool) -> transforms.Compose:
        if augment:
            transforms_list = [
                transforms.RandomCrop(  # random pad image to image_shape with 0
                    size=image_shape, padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
                ),
                transforms.ColorJitter(brightness=(0.8, 1.6)),
                transforms.RandomAffine(
                    degrees=1,
                    shear=(-10, 10),
                    resample=Image.BILINEAR,
                ),
            ]
        else:
            transforms_list = [transforms.CenterCrop(image_shape)]  # pad image to image_shape with 0
        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)

def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample=Image.BILINEAR)


if __name__ == "__main__":
    load_and_print_info(MNIST)

