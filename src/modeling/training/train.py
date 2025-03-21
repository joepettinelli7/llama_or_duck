import logging
import math
import sys
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.modeling.training import utils
from src.modeling.training.config import EARLY_STOP_PATIENCE
from src.modeling.model.mobilenet_v2 import CustomMobileNetV2


def evaluate(model: CustomMobileNetV2,
             loss_func: torch.nn.BCEWithLogitsLoss,
             data_loader: torch.utils.data.DataLoader,  # ignore
             device: torch.device,
             epoch: int,
             logger: logging.Logger) -> float:
    """
    Runs inference with the provided model and dataset and logs the computed metrics.

    Args:
        model: CustomMobileNetV2 to be trained.
        loss_func: The loss function object.
        data_loader: Train validation dataset to be used.
        device: Evaluates on CPU or GPU.
        epoch: Current epoch.
        logger: The logger.

    Returns:
        Return the val loss from metric logger
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", mode="val")
    header = f"Epoch: [{epoch}]-Validation"
    # Don't compute gradients during evaluation
    with torch.no_grad():
        for images_batch, targets_batch in metric_logger.log_every(data_loader, header, logger):
            images_batch.to(device)
            model_output = model(images_batch)
            loss_tensor = loss_func(model_output.squeeze(), targets_batch)
            loss_value: float = loss_tensor.item()
            metric_logger.update(loss=loss_value)
    val_loss = metric_logger.meters['loss'].median
    return val_loss


def train_one_epoch(model: CustomMobileNetV2,
                    optimizer: torch.optim.Optimizer,
                    loss_func: torch.nn.BCEWithLogitsLoss,
                    data_loader: torch.utils.data.DataLoader,  # ignore
                    device: torch.device,
                    epoch: int,
                    logger: logging.Logger) -> None:
    """
    Train the model for one epoch.

    Args:
        model: CustomMobileNetV2 model to be trained.
        optimizer: Optimizer to be used during training. Usually Adam.
        loss_func: The loss function object.
        data_loader: Train dataset to be used for training.
        device: Train on CPU or GPU.
        epoch: Current epoch.
        logger: The logger.

    Returns:

    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]-train".format(epoch)
    lr_scheduler = None
    if epoch == 1:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    for images_batch, targets_batch in metric_logger.log_every(data_loader, header, logger):
        images_batch.to(device)
        model_output = model(images_batch)
        loss_tensor = loss_func(model_output.squeeze(), targets_batch.float())
        loss_value: float = loss_tensor.item()
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            logger.info(loss_value)
            sys.exit(1)
        optimizer.zero_grad()  # flush gradients memory
        loss_tensor.backward()  # compute gradients
        optimizer.step()  # adjust parameters
        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def train_model(dataloaders: [str, torch.utils.data.DataLoader],  # ignore
                model: CustomMobileNetV2,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: torch.optim.lr_scheduler.StepLR,
                loss_func: torch.nn.BCEWithLogitsLoss,
                num_epochs: int,
                device: torch.device,
                logger: logging.Logger) -> CustomMobileNetV2:
    """
    Train the model over all epochs.

    Args:
        dataloaders: Train and validation dataset to be used for training.
        model: CustomMobileNetV2 network to be trained.
        optimizer: Optimizer to be used during training. Usually Adam.
        lr_scheduler: Scheduler for learning rate.
        loss_func: The loss function object
        num_epochs: Number of epochs to be trained.
        device: Train on CPU or GPU.
        logger: Log info

    Returns:
        Trained model
    """
    since = time.time()
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, delta=0.0)
    epoch_loss_values: list[float] = []
    for epoch in range(num_epochs):
        # Train for one epoch
        print(f'{epoch + 1} / {num_epochs}')
        train_one_epoch(model,
                        optimizer,
                        loss_func,
                        dataloaders["train"],
                        device,
                        epoch + 1,
                        logger=logger)
        # Update the learning rate
        lr_scheduler.step()
        # Evaluate model
        validation_loss: float = evaluate(model,
                                          loss_func,
                                          dataloaders["val"],
                                          device,
                                          epoch + 1,
                                          logger=logger)
        epoch_loss_values.append(validation_loss)
        # Potentially stop training if validation loss increased
        should_stop = early_stopping(validation_loss)
        if should_stop:
            print("Early stopping triggered!")
            break
    time_elapsed = time.time() - since
    logging.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    # Plot loss values
    plt.plot(list(range(0, len(epoch_loss_values))), epoch_loss_values)
    plt.title("BCE Loss Trend Over Increasing Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('loss_plot.png')
    plt.clf()
    return model


class EarlyStopping:

    def __init__(self, patience: int, delta: float) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None

    def __call__(self, score: float) -> bool:
        """
        Determine if training should stop.
        Stop if validation loss increases.

        Args:
            score: The training loss

        Returns:
            True, if it should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


if __name__ == "__main__":
    pass
