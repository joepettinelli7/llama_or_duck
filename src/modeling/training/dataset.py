import os
from typing import Dict
from typing import Tuple
import torch
import torch.utils.data
from PIL import Image
import shutil

from src.modeling.training.config import VALIDATION_SPLIT
from src.modeling.image_processing import ImageProcessor


class ODDataset(torch.utils.data.Dataset):
    """
    Construct a dataset object from the dataset directory.
    """

    def __init__(self, train_dir: str) -> None:
        """
        Create an object for the image classification dataset.

        Args:
            train_dir: Directory where training data is.

        Returns:

        """
        # Get sorted files
        self.img_paths: list[str] = self.get_image_paths(train_dir)
        self.class_to_idx: dict[str, int] = {"llama": 1, "duck": 0}
        self.img_processor = ImageProcessor(for_training=True)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the image with label.

        Args:
            idx: Image index in training data.

        Returns:
            The image tensor with info.
        """
        # Load image and annotation which is just label
        img_path = self.img_paths[idx]
        img: Image = Image.open(img_path)
        # Preprocess image expected by model
        input_tensor = self.img_processor.preprocess_for_model(img)
        # Assign label
        label = 1 if "llama" in img_path else 0
        # Use float32 for torch.nn.BCEWithLogitsLoss()
        label_tensor = torch.as_tensor(label, dtype=torch.float32)
        return input_tensor, label_tensor

    def __len__(self) -> int:
        """
        Need this in subclass.

        Returns:
            The size of dataset.
        """
        return len(self.img_paths)

    @staticmethod
    def get_image_paths(train_dir: str) -> list[str]:
        """
        Get paths of all images in model_input directory.

        Args:
            train_dir: Directory where training data is.

        Returns:
            List of image paths.
        """
        duck_path = os.path.join(train_dir, "duck")
        llama_path = os.path.join(train_dir, "llama")
        duck_image_paths = [os.path.join(duck_path, file) for file in os.listdir(duck_path) if "._" not in file]
        llama_image_paths = [os.path.join(llama_path, file) for file in os.listdir(llama_path) if "._" not in file]
        image_paths = sorted(duck_image_paths + llama_image_paths)
        print(f'Dataloader found {len(image_paths)} input images.')
        return image_paths


def make_replicates(train_dir: str) -> None:
    """
    Make replicate that has 75% to be augmented later. There
    is 94% chance that either the original or replicate is augmented.
    Do this to increase number and variability in training data

    Args:
        train_dir: The training directory

    Returns:

    """
    print("Making images replicates...")
    image_paths = ODDataset.get_image_paths(train_dir)
    for image_path in image_paths:
        base_path, ext = os.path.splitext(image_path)
        replicate_path = f"{base_path}_rep{ext}"
        shutil.copy(image_path, replicate_path)


def delete_replicates(train_dir: str) -> None:
    """
    Delete replicates from train_dir

    Args:
        train_dir: The training directory

    Returns:

    """
    image_paths = ODDataset.get_image_paths(train_dir)
    n_removed: int = 0
    for image_path in image_paths:
        if "_rep" in image_path:
            os.remove(image_path)
            n_removed += 1
    print(f"Deleted {n_removed} replicates.")


def prepare_dataloader(train_dir: str, batch_size: int) -> Tuple[Dict[str, torch.utils.data.DataLoader],
                                                                 Dict[str, int]]:
    """
    Create and return train/validation dataloaders,
    train/validation sizes.

    Args:
        train_dir: Directory where training dataset is.
        batch_size: size of batch in dataloader.

    Returns:
        dataloaders: {'train': torch.utils.DataLoader, 'val': torch.utils.DataLoader}
        dataset_sizes: {'train':size of train dataset, 'val': size of validation dataset}
    """
    # Initialize data loaders
    make_replicates(train_dir)
    dataset = ODDataset(train_dir)
    val_dataset = ODDataset(train_dir)
    torch.manual_seed(2024)
    indices = torch.randperm(len(dataset)).tolist()
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "val": torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,
        ),
    }
    dataset_sizes = {"train": len(dataset) -
                     val_size, "val": val_size}
    return dataloaders, dataset_sizes


if __name__ == "__main__":
    pass
