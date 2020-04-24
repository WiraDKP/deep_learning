import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from jcopdl.callback import Callback, set_config

from src.model import CNN
from src.train_utils import loop_fn

import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    # Dataset & Dataloader
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(cfg.CROP_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(70),
        transforms.CenterCrop(cfg.CROP_SIZE),
        transforms.ToTensor()
    ])

    train_set = datasets.ImageFolder(cfg.TRAIN_DIR, transform=train_transform)
    trainloader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)

    test_set = datasets.ImageFolder(cfg.TEST_DIR, transform=test_transform)
    testloader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # Config
    config = set_config({
        "batch_size": cfg.BATCH_SIZE,
        "crop_size": cfg.CROP_SIZE
    })
    
    # Training Preparation
    model = CNN().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    callback = Callback(model, config, outdir=cfg.OUTDIR)

    # Training
    while True:
        train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
        with torch.no_grad():
            test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)

        # Callbacks
        callback.log(train_cost, test_cost, train_score, test_score)
        callback.save_checkpoint()
        if callback.early_stopping(model, monitor="test_score"):
            break
            
if __name__ == "__main__":
    train()