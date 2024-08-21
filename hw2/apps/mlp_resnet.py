import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = (
        [nn.Linear(dim, hidden_dim), nn.ReLU()]
        + [
            ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
            for _ in range(num_blocks)
        ]
        + [nn.Linear(hidden_dim, num_classes)]
    )
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = nn.SoftmaxLoss()
    batch_loss, err_samples, num_samples = ([], 0, 0)

    if opt:
        model.train()
        for idx, batch in enumerate(dataloader):
            print(batch[0].shape)
            opt.reset_grad()
            X, y = batch
            bsz = X.shape[0]
            X = X.reshape((bsz, -1))
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            prediction = logits.numpy().argmax(axis=-1)
            errs = prediction != y.numpy()
            err_samples += errs.sum()
            num_samples += errs.shape[0]
            batch_loss.append(loss.numpy())
    else:
        model.eval()
        for idx, batch in enumerate(dataloader):
            X, y = batch
            bsz = X.shape[0]
            X = X.reshape((bsz, -1))
            logits = model(X)
            loss = loss_fn(logits, y)
            prediction = logits.numpy().argmax(axis=-1)
            errs = prediction != y.numpy()
            err_samples += errs.sum()
            num_samples += errs.shape[0]
            batch_loss.append(loss.numpy())

    return err_samples / num_samples, np.mean(batch_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    resnet = MLPResNet(28 * 28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    test_set = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    train_loader = ndl.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, resnet, opt)
    test_err, test_loss = epoch(test_loader, resnet, None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
