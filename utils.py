import matplotlib.pyplot as plt
import numpy as np
import ray.train.torch
import torch
import torchvision
from filelock import FileLock
from ray import train
from torch import nn, optim
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision import models

###############################################################################
# Visualization and Prediction
###############################################################################

CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck')


def get_class(id):
    return CLASSES[id]


def show_images(images, labels):
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    imshow(torchvision.utils.make_grid(images))
    print("Labels: ", labels)
    print("Labels: ",
          " ".join(f"{get_class(label):5s}" for label in labels))


def predict(model, images):
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    print("Predictions: ", predictions)
    print("Predictions: ",
          " ".join(f"{get_class(prediction):5s}" for prediction in predictions))
    return predictions


def predict_and_display(model, images, labels):
    show_images(images, labels)
    print()
    predictions = predict(model, images)
    print()
    num_correct = predictions.eq(labels).sum().item()
    num_total = len(images)
    print(
        f"Predicted {num_correct}/{num_total} correctly. "
        f"Accuracy: {num_correct / num_total:.0%}.")


###############################################################################
# Data Preparation
###############################################################################

def get_dataset(train: bool):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    data_transform = train_transform if train else test_transform
    with FileLock(".ray.lock"):
        return datasets.CIFAR10(root="data", train=train,
                                transform=data_transform, download=train)


def get_test_data(num_records):
    test_dataset = get_dataset(False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=num_records,
                                                  shuffle=True)
    images, labels = next(iter(test_dataloader))
    return (images, labels)


###############################################################################
# Model Preparation
###############################################################################

def get_model(state_dict=None):
    model = models.resnet50(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = torch.nn.Identity()
    if state_dict:
        model.load_state_dict(state_dict)
    return model


def get_state_dict(model):
    model_state_dict = model.state_dict()
    consume_prefix_in_state_dict_if_present(model_state_dict, "module.")
    return model_state_dict


###############################################################################
# Hyperparameter Preparation
###############################################################################

def get_config(config=None):
    config = config or {}
    config.setdefault("num_epochs", 50)
    config.setdefault("batch_size", 32)
    config.setdefault("lr", 0.01)
    config.setdefault("max_lr", 0.8)
    config.setdefault("momentum", 0.9)
    config.setdefault("weight_decay", 0.0001)
    config.setdefault("test_mode", False)
    return config


###############################################################################
# Training and Validation
###############################################################################

def train_epoch(train_loader, model, criterion, optimizer, scheduler):
    running_loss = 0
    num_images = 0

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        batch_size = len(images)
        num_images += batch_size

        output = model(images)

        loss = criterion(output, labels)
        running_loss += loss.item() * batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    epoch_loss = running_loss / num_images
    results = {"train_loss": epoch_loss}
    return results


def validate_epoch(val_loader, model, criterion):
    model.eval()
    running_loss = 0
    num_images = 0
    num_correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            batch_size = len(images)
            output = model(images)
            loss = criterion(output, labels)

            _, predicted = torch.max(output.data, 1)
            num_images += len(images)

            num_correct += predicted.eq(labels).sum().item()
            running_loss += loss.item() * batch_size

    epoch_loss = running_loss / num_images
    accuracy = num_correct / num_images
    results = {"val_loss": epoch_loss, "val_accuracy": accuracy}
    return results


###############################################################################
# Training Loop
###############################################################################

def train_func(config):
    config = get_config(config)
    print(f"Config: {config}")

    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    max_lr = config["max_lr"]
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]
    test_mode = config["test_mode"]

    worker_batch_size = batch_size
    # CHANGE 1: Split global batch size amongst workers.
    worker_batch_size = batch_size // train.world_size()
    print(f"World size: {train.world_size()}. ",
          f"Worker batch size: {worker_batch_size}.")

    # Load model.
    model = get_model()
    # CHANGE 2: Distribute your model.
    model = train.torch.prepare_model(model)

    # Get training and validation dataloaders.
    train_dataset = get_dataset(train=True)
    val_dataset = get_dataset(train=False)
    if test_mode:
        train_dataset = Subset(train_dataset, list(range(64)))
        val_dataset = Subset(val_dataset, list(range(64)))
    train_loader = DataLoader(train_dataset, batch_size=worker_batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=worker_batch_size)
    # CHANGE 3: Shard your data.
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)

    # Define loss function and optimizer and scheduler.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                              steps_per_epoch=len(train_loader),
                                              epochs=num_epochs)

    results = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch} | Start")
        train_results = train_epoch(train_loader, model, criterion, optimizer,
                                    scheduler)
        print(f"Epoch {epoch} | Training results: {train_results}")
        validation_results = validate_epoch(val_loader, model, criterion)
        print(f"Epoch {epoch} | Validation results: {validation_results}")

        model_state_dict = get_state_dict(model)
        optimizer_state_dict = optimizer.state_dict()
        results = {**train_results, **validation_results,
                   "model_state_dict": model_state_dict}

        # CHANGE 4: Report metrics.
        train.report(epoch=epoch, lr=scheduler.get_last_lr()[0],
                     **train_results, **validation_results)

        # CHANGE 5: Enable checkpointing.
        train.save_checkpoint(epoch=epoch,
                              accuracy=validation_results["val_accuracy"],
                              model_state_dict=model_state_dict,
                              optimizer_state_dict=optimizer_state_dict
                              )

#     return results
