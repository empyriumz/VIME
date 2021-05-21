import torch
import torch.nn as nn
from vime_utils import unsup_generator

def train_unsup(args, model, device, train_loader, optimizer, epoch):
    train_loss = 0
    model.train()
    recon_loss_cont = nn.MSELoss()
    recon_loss_cat = nn.BCELoss()
    mask_loss = nn.BCEWithLogitsLoss()
    for batch_idx, (data, _) in enumerate(train_loader):
        # data = data.view(-1, 784)
        x, x_tilde, mask, cont_col, cat_col = unsup_generator(args.pm, data, device)
        optimizer.zero_grad()
        _, mask_recon, x_recon = model(mask, x_tilde)
        x_recon_cont = x_recon[:, cont_col]
        x_recon_cat = x_recon[:, cat_col]
        x_cont = x[:, cont_col]
        x_cat = x[:, cat_col]
        loss = mask_loss(mask_recon, mask) + args.alpha * (
            recon_loss_cont(x_recon_cont, x_cont)
            + recon_loss_cat(x_recon_cat, x_cat)
        )
        train_loss += len(data) * loss.item()
        #train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break
    train_loss /= len(train_loader.sampler)
    print("Average train loss for epoch {}: {:.4f}\n".format(epoch, train_loss))
    return train_loss

def train_sup(args, model, device, train_loader, optimizer, epoch):
    train_loss = 0
    model.train()
    recon_loss_cont = nn.MSELoss()
    recon_loss_cat = nn.BCELoss()
    mask_loss = nn.BCEWithLogitsLoss()
    for batch_idx, (data, _) in enumerate(train_loader):
        x, x_tilde, mask, cont_col, cat_col = unsup_generator(args.pm, data, device)
        optimizer.zero_grad()
        _, mask_recon, x_recon = model(mask, x_tilde)
        x_recon_cont = x_recon[:, cont_col]
        x_recon_cat = x_recon[:, cat_col]
        x_cont = x[:, cont_col]
        x_cat = x[:, cat_col]
        loss = mask_loss(mask_recon, mask) + args.alpha * (
            recon_loss_cont(x_recon_cont, x_cont)
            + recon_loss_cat(x_recon_cat, x_cat)
        )
        train_loss += len(data) * loss.item()
        #train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break
    train_loss /= len(train_loader.sampler)
    print("Average train loss for epoch {}: {:.4f}\n".format(epoch, train_loss))
    return train_loss

def test_unsup(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    recon_loss_cont = nn.MSELoss()
    recon_loss_cat = nn.BCELoss()
    mask_loss = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for data, _ in test_loader:
            x, x_tilde, mask, cont_col, cat_col = unsup_generator(args.pm, data, device)
            _, mask_recon, x_recon = model(mask, x_tilde)
            x_recon_cont = x_recon[:, cont_col]
            x_recon_cat = x_recon[:, cat_col]
            x_cont = x[:, cont_col]
            x_cat = x[:, cat_col]
            loss = mask_loss(mask_recon, mask) + args.alpha * (
                    recon_loss_cont(x_recon_cont, x_cont)
                + recon_loss_cat(x_recon_cat, x_cat)
                )
            test_loss += len(data) * loss.item()

    test_loss /= len(test_loader.sampler)

    print("\nTest set: Average loss: {:.4f}\n".format(test_loss,))
    return test_loss