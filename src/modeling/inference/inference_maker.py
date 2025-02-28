import torch
from PIL import Image

from modeling.model.mobilenet_v2 import create_model
from modeling.image_processing import ImageProcessor


class InferenceMaker:

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = create_model(device, for_training=False)
        self.model.eval()
        self.img_processor = ImageProcessor(for_training=False)

    def infer(self, image: Image) -> bool:
        """
        Make inference on image using model.

        Args:
            image: Image to make inference on.

        Returns:
            True if llama else False for duck.
        """
        input_tensor: torch.Tensor = self.img_processor.preprocess_for_model(image)
        input_tensor.to(self.device)
        with torch.no_grad():
            output_logit = self.model(input_tensor.unsqueeze(0))
            probability = torch.sigmoid(output_logit)
            print(round(probability.item(), 2))
            is_llama = probability.item() > 0.5
            return is_llama


if __name__ == "__main__":
    pass
