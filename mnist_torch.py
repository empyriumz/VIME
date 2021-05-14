from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from vime_utils import unsup_data_generator, EarlyStopping
from data_loader import mnist_datasets
from vime_self import VIME_Encoder

def train(args, model, device, train_loader, optimizer, epoch):
    train_loss = 0
    model.train()
    recon_loss = nn.MSELoss()
    mask_loss= nn.BCEWithLogitsLoss()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784)
        mask, x_tilde = unsup_data_generator(args.pm, data, device)
        optimizer.zero_grad()
        _, mask_recon, x_recon = model(mask, x_tilde)
        loss = mask_loss(mask_recon, mask) + args.alpha * recon_loss(x_recon, data.to(device))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )
            if args.dry_run:
                break
    train_loss /= len(train_loader.dataset)
    print(
        "Average train loss for epoch {}: {:.4f}\n".format(epoch, 
            train_loss,
        )
    )
    return train_loss

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    recon_loss = nn.MSELoss()
    mask_loss= nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, 784)
            mask, x_tilde = unsup_data_generator(args.pm, data, device)
            mask, x_tilde = mask.to(device), x_tilde.to(device)
            _, mask_recon, x_recon = model(mask, x_tilde)
            loss = mask_loss(mask_recon, mask) + args.alpha * recon_loss(x_recon, data.to(device))
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}\n".format(
            test_loss,
        )
    )
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        metavar="N",
        help="input batch size for training (default: 512)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        metavar="N",
        help="hidden dimensions",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        metavar="LR",
        help="learning rate (default: 0.5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        metavar="Alpha",
        help="ratio between reconstruction loss and mask loss",
    )
    parser.add_argument(
        "--pm",
        type=float,
        default=0.2,
        help="Ratio of masked data",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader, test_loader = mnist_datasets(args)
    input_dim = 784
    model = VIME_Encoder(input_dim, args.hidden_dim).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    early_stopping = EarlyStopping(verbose=True, path="./save_model/mnist_encoder.pt")
    train_loss = []
    valid_loss = []
    for epoch in range(1, args.epochs + 1):      
        loss = train(args, model, device, train_loader, optimizer, epoch)
        train_loss.append(loss)
        val_loss = test(args, model, device, valid_loader)
        valid_loss.append(val_loss)
        scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    main()
