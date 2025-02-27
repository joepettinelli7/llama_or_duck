import typing
from PIL import Image
import numpy as np
import torch
from torchvision import transforms


class ImageProcessor:

    def __init__(self, for_training: bool = False) -> None:
        self.img_transforms = self.get_transforms(for_training=for_training)

    @staticmethod
    def get_transforms(for_training: bool = False) -> transforms.Compose:
        """
        Get the transforms for images.

        1. Resize to about minimum size for model.
        2. Transform to tensor.
        3. Normalize.
        4. Random horizontal flip if for training.
        5. Random vertical flip if for training.

        Args:
            for_training: Whether getting transforms for training.

        Returns:
            The list of transforms to apply to images.
        """
        transform_list = [
            transforms.Resize((int(680 / 2.5), int(580 / 2.5))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if for_training:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.50))
            transform_list.append(transforms.RandomVerticalFlip(p=0.50))
        return transforms.Compose(transform_list)

    def preprocess_for_model(self, img: typing.Union[np.ndarray, Image]) -> torch.Tensor:
        """
        Preprocess the image for model. Use this function
        as single source so ensure same processing is done for
        training and inference.

        Args:
            img: The image to preprocess. Accept numpy array or PIL Image, but
                 PIL Image is needed by torch.

        Returns:
            The tensor for model input.
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        assert img.mode == 'RGB' and img.size == (580, 680)
        return self.img_transforms(img)


if __name__ == "__main__":
    pass
