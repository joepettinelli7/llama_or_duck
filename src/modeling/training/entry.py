import argparse
import logging
import sys
import typing
import torch

from src.modeling.training.config import TRAIN_DIR
from src.modeling.model.mobilenet_v2 import CustomMobileNetV2, create_model
from src.modeling.training.dataset import prepare_dataloader, delete_replicates
from src.modeling.training.train import train_model
from src.modeling.training import config

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


def _parse_args():
    """
    Parse arguments / config variables.

    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--step-size", type=int, default=config.DEFAULT_STEP_SIZE)
    parser.add_argument("--gamma", type=float, default=config.DEFAULT_GAMMA)
    parser.add_argument("--train-only-top-layer", type=str, default=config.TRAIN_ONLY_TOP_LAYER)
    return parser.parse_known_args()


def get_top_layer_param_names(model: CustomMobileNetV2) -> typing.List[str]:
    """
    Return the names of parameters of the classifier layer.

    Args:
        model: The model being used for training.

    Returns:
        Names of classifier layer params.
    """
    top_layer_param_names = [param_name for param_name, _ in model.named_parameters() if "classifier" in param_name]
    return top_layer_param_names


def set_requires_grad(model: typing.Any, train_only_top_layer: bool) -> None:
    """
    Set the require_grad attribute for each parameter based on train_only_top_layer.

    Parameters in the last layer are always trainable (requires_grad = True).
    For rest of the parameters, requires_grad = not train_only_top_layer.

    Args:
        model: The model being used.
        train_only_top_layer: Whether to train only top layer.

    Returns:

    """
    top_layer_param_names = get_top_layer_param_names(model)
    for param_name, param in model.named_parameters():
        if param_name not in top_layer_param_names:
            param.requires_grad = not train_only_top_layer
        else:
            param.requires_grad = True


def run_with_args(args) -> None:
    """
    Run training.

    Returns:

    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Training on {device}.")
    dataloaders, dataset_sizes = prepare_dataloader(TRAIN_DIR, batch_size=args.batch_size)
    logging.info(f"Dataset sizes: {dataset_sizes}")
    model: CustomMobileNetV2 = create_model(device, for_training=True)
    set_requires_grad(model, train_only_top_layer=args.train_only_top_layer)
    params = [p for p in model.parameters() if p.requires_grad]
    # Adaptive moment estimation
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    # Learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # Binary cross entropy combined with sigmoid activation. Pass raw logits.
    loss_func = torch.nn.BCEWithLogitsLoss()
    print('Start training')
    trained_model: CustomMobileNetV2 = train_model(
        dataloaders=dataloaders,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_func=loss_func,
        num_epochs=args.epochs,
        device=device,
        logger=logger
    )
    # Save model weights
    torch.save(trained_model.state_dict(), "../model/model_inf_weights.pth")
    # Delete replicated images
    delete_replicates(TRAIN_DIR)
    print("Saved weights to model directory.")


if __name__ == "__main__":
    _args, unknown = _parse_args()
    run_with_args(_args)
