from src.get_mnist import load
from src.mnist_dataset import Mnist
from src.lenet5 import LeNet5
from src.train import train

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import logging
import matplotlib.pyplot as plt
import numpy as np 


logging.basicConfig(
        filename="logs/main.log",
        encoding="utf-8", 
        filemode="a", 
        format="{asctime}-{levelname}-{message}",
        style="{", 
        datefmt="%d/%m/%Y %H:%M", 
        level=logging.DEBUG
)
logging.getLogger('matplotlib.font_manager').disabled = True


def main() -> None: 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using {device}")

    batch_size = 1
    epochs = 1

    (train_images, train_labels), (test_images, test_labels) = load()  
    train_dataset = Mnist(train_images, train_labels)
    test_dataset = Mnist(test_images, test_labels)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1], generator)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)

    model = LeNet5().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    train_loss, val_loss = train(train_dataloader, val_dataloader, 
            model, loss_fn, optimizer, epochs, device)

    fig = plt.figure(figsize=(14,5))
    plt.plot(np.arange(epochs), train_loss, label='train')
    plt.plot(np.arange(epochs), val_loss, label='val')
    plt.legend()
    plt.savefig("figures/train_val_loss")
    

    

if __name__ == "__main__":
    main()