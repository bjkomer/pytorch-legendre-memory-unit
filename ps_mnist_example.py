# based on:
# https://github.com/abr/neurips2019/blob/master/experiments/psMNIST-standard.ipynb
import numpy as np
import os
import sys
from lmu import LMUCell
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import accuracy_score
from functools import partial

from tensorboardX import SummaryWriter
from datetime import datetime

nru_path = os.path.abspath("nru/nru_project")
sys.path.append(nru_path)

from utils.utils import create_config
from train import get_data_iterator


class Dataset(data.Dataset):

    def __init__(self, inputs, outputs, return_list=True):

        self.inputs = inputs.astype(np.float32)
        self.outputs = outputs.astype(np.long)

        # flag for whether the inputs returned are a single tensor or a list of tensors
        self.return_list = return_list

    def __getitem__(self, index):

        if self.return_list:
            return [self.inputs[index, i] for i in range(self.inputs.shape[1])], \
                   self.outputs[index],
        else:
            return self.inputs[index], self.outputs[index]

    def __len__(self):
        return self.inputs.shape[0]


class LMU(nn.Module):
    def __init__(self, units=212, order=256, theta=28**2):
        super(LMU, self).__init__()

        self.theta = theta
        self.units = units
        self.order = order

        self.lmu_cell = LMUCell(
            input_size=1,
            hidden_size=units,
            order=order,
            input_encoders_initializer=partial(torch.nn.init.constant_, val=1),
            hidden_encoders_initializer=partial(torch.nn.init.constant_, val=0),
            memory_encoders_initializer=partial(torch.nn.init.constant_, val=0),
            input_kernel_initializer=partial(torch.nn.init.constant_, val=0),
            hidden_kernel_initializer=partial(torch.nn.init.constant_, val=0),
            memory_kernel_initializer=torch.nn.init.xavier_normal_,
        )

        self.dense = nn.Linear(
            in_features=units,
            out_features=10,
        )
        self.softmax = nn.Softmax()

    def forward(self, inputs):

        # TODO: is this the correct hidden state initialization?
        h = torch.zeros(1, self.units)
        c = torch.zeros(1, self.order)

        for input in inputs:
            h, c = self.lmu_cell(input, (h, c))

        return self.softmax(self.dense(h), dim=1)


def evalutate(model, dataloader, epoch, batch_size=100, writer=None, name='validation'):
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        avg_loss = 0
        avg_acc = 0
        n_batches = 0
        for i, data in enumerate(dataloader):
            inputs, outputs = data

            if outputs.size()[0] != batch_size:
                continue  # Drop data, not enough for a batch

            pred = model(inputs)

            loss = criterion(pred, outputs)

            avg_loss += loss.data.item()

            avg_acc += accuracy_score(outputs.detach().numpy(), np.argmax(pred.detach().numpy(), axis=1))

            n_batches += 1

        avg_loss /= n_batches
        avg_acc /= n_batches
        print("{} loss:".format(name), avg_loss)
        print("{} acc:".format(name), avg_acc)

        if writer is not None:
            writer.add_scalar('avg_{}_loss'.format(name), avg_loss, epoch + 1)
            writer.add_scalar('avg_{}_accuracy'.format(name), avg_acc, epoch + 1)


def get_ps_mnist():

    try:
        cwd = os.getcwd()
        os.chdir(nru_path)  # needed to load parent config/default.yaml
        os.environ["PROJ_SAVEDIR"] = "/tmp/"  # this shouldn't do anything
        config = create_config(os.path.join(nru_path, "config/nru.yaml"))
    finally:
        os.chdir(cwd)

    padded_length = 785
    batch_size = 100

    mask_check = np.zeros((padded_length, batch_size))
    mask_check[-1, :] = 1

    from collections import defaultdict
    X = defaultdict(list)
    Y = defaultdict(list)

    gen = get_data_iterator(config)  # uses a fixed data seed
    for tag in ("train", "valid", "test"):
        while True:
            data = gen.next(tag)
            if data is None:
                break

            assert data['x'].shape == (padded_length, batch_size, 1)
            assert data['y'].shape == data['mask'].shape == (padded_length, batch_size)
            assert np.all(data['mask'] == mask_check)

            assert np.all(data['x'][-1, :, :] == 0)
            X[tag].extend(data['x'][:-1, :, :].transpose(1, 0, 2))

            assert np.all(data['y'][:-1, :] == 0)
            Y[tag].extend(data['y'][-1, :])

    X_train = np.asarray(X["train"])
    X_valid = np.asarray(X["valid"])
    X_test = np.asarray(X["test"])

    Y_train = np.asarray(Y["train"])
    Y_valid = np.asarray(Y["valid"])
    Y_test = np.asarray(Y["test"])

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def main(batch_size=100, n_epochs=10, logdir='ps_mnist_lmu'):

    # Load the dataset
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_ps_mnist()

    print(X_train.shape, Y_train.shape)
    print(X_valid.shape, Y_valid.shape)
    print(X_test.shape, Y_test.shape)

    # Create the PyTorch data loaders
    trainset = Dataset(
        inputs=X_train,
        outputs=Y_train,
    )
    validset = Dataset(
        inputs=X_valid,
        outputs=Y_valid,
    )
    testset = Dataset(
        inputs=X_test,
        outputs=Y_test,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    # Set up tensorboard writer
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = os.path.join(logdir, current_time)
    writer = SummaryWriter(log_dir=save_dir)

    # Define the LMU model
    model = LMU(units=212, order=256, theta=28**2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print("Training")
    for epoch in range(n_epochs):
        print('Epoch {} of {}'.format(epoch + 1, n_epochs))

        avg_loss = 0
        avg_acc = 0
        n_batches = 0
        for i, data in enumerate(trainloader):
            inputs, outputs = data

            if outputs.size()[0] != batch_size:
                continue  # Drop data, not enough for a batch
            optimizer.zero_grad()

            pred = model(inputs)

            loss = criterion(pred, outputs)

            loss.backward()

            avg_loss += loss.data.item()

            avg_acc += accuracy_score(outputs.detach().numpy(), np.argmax(pred.detach().numpy(), axis=1))

            optimizer.step()

            n_batches += 1

        avg_loss /= n_batches
        avg_acc /= n_batches
        print("train loss:", avg_loss)
        print("train acc:", avg_acc)

        writer.add_scalar('avg_train_loss', avg_loss, epoch + 1)
        writer.add_scalar('avg_train_accuracy', avg_acc, epoch + 1)

        evalutate(model, validloader, epoch=epoch+1, batch_size=batch_size, writer=writer, name='validation')

    print("Testing")
    evalutate(model, testloader, epoch=n_epochs, batch_size=batch_size, writer=writer, name='test')

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))


if __name__ == '__main__':
    # Run the Permuted Sequential MNIST Example
    main()
