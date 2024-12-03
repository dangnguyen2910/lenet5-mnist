from get_mnist import load
from mnist_dataset import Mnist
from model import LeNet5
from train import train

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import logging



def main() -> None: 
    logging.basicConfig(
        filename="logs/main.log",
        encoding="utf-8", 
        filemode="a", 
        format="{asctime}-{levelname}-{message}",
        style="{", 
        datefmt="%d/%m/%Y %H:%M", 
        level=logging.DEBUG
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using {device}")

    batch_size = 1
    epochs = 2

    (train_images, train_labels), (test_images, test_labels) = load()  
    train_dataset = Mnist(train_images, train_labels)
    test_dataset = Mnist(test_images, test_labels)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1], generator)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    model = LeNet5().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trained_model = train(train_dataloader, val_dataloader, 
                          model, loss_fn, optimizer, epochs, device)
    

    

if __name__ == "__main__":
    main()