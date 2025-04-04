from collections import OrderedDict
from typing import Optional, List, Callable
import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class CustomInvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class CustomMobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            num_classes: Number of classes
            width_mult: Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest: Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout: The dropout probability

        """
        super().__init__()

        if block is None:
            block = CustomInvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = self.make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = self.make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = self.make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @staticmethod
    def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
        """
        It ensures that all layers have a channel number that is divisible by 8

        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model. This exists since TorchScript doesn't
        support inheritance, so the superclass method (this one) needs to have a name
        other than `forward` that can be accessed in a subclass

        Args:
            x: The image tensor

        Returns:

        """
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Call _forward where actual work is done

        """
        return self._forward(x)


def remap_weight_keys(pretrained_weights: OrderedDict, model_state_dict: dict) -> dict:
    """
    Remap pytorch weights to be compatible with CustomMobileNetV2
    which only has 1 class compared to pretrained 1000. To do this,
    use untrained weights in classifier layer:

    (classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=1280, out_features=1, bias=True)
  )

    Args:
        pretrained_weights: Pretrained weights from pytorch
        model_state_dict: Weights from CustomMobileNetV2

    Returns:
         pytorch weights with CustomResnet keys()
    """
    exclude_weight_keys = ['classifier.1.weight',  'classifier.1.bias']
    pretrained_weight_keys = list(pretrained_weights.keys())
    for pretrained_weight_key in pretrained_weight_keys:
        if pretrained_weight_key in exclude_weight_keys:
            print(f"Not using '{pretrained_weight_key}' from pretrained weights.")
            pretrained_weights[pretrained_weight_key] = model_state_dict[pretrained_weight_key]
        else:
            pass
    return pretrained_weights


def create_model(device: torch.device, for_training: bool = False) -> CustomMobileNetV2:
    """
    Create the model object

    Args:
        device: The device cpu or gpu
        for_training: Whether model is for training or not.
                        If True then use pretrained weights. If False
                        use inference weights

    Returns:
         The model. Does not set to eval mode.
    """
    n_classes = 1
    custom_model = CustomMobileNetV2(num_classes=n_classes)
    custom_model_sd = custom_model.state_dict()
    if for_training:
        pretrain_weights = torch.load("../model/mobilenet_v2-b0353104.pth")
        weights = remap_weight_keys(pretrain_weights, custom_model_sd)
    else:
        weights = torch.load("./modeling/model/model_inf_weights.pth")
    custom_model.load_state_dict(weights)
    custom_model.to(device)
    return custom_model


if __name__ == "__main__":
    pass
