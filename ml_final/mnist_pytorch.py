from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

MNIST_TRAIN_SIZE = 55000
MNIST_TEST_SIZE = 10000
# SKLEARN_DIGITS_TRAIN_SIZE = 1247
# SKLEARN_DIGITS_TEST_SIZE = 550


# def get_mnist_dataset(loader):  # pragma: no cover
#     """Downloads MNIST as PyTorch dataset.
#
#     Parameters
#     ----------
#     loader : str (values: 'train' or 'test')."""
#     dataset = datasets.MNIST(
#         root="../data",
#         train=(loader == "train"),
#         download=True,
#         transform=transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#         ),
#     )
#     return dataset


def Myloader(path):
    return Image.open(path).convert('L')


# get a list of paths and labels.
def init_process(path1, path2):
    i = 0
    data = []
    with open(path1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            a = line.split()
            x = a[1]
            i=int(a[0])
            data.append([path2 % i, x])
    return data


class mydataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def load_data():
    print('data processing...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # normalization
    ])
    path1 = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\final_train_labs.txt'
    path2 = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\train\\%d.png'
    data1 = init_process(path1, path2)
    path3 = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\test_labs.txt'
    path4 = 'D:\\p_y\\dataset\\MNIST\\mnist_dataset\\test\\%d.png'
    data2 = init_process(path3, path4)
    # np.random.shuffle(data1)
    # np.random.shuffle(data2)
    train_data, test_data = data1, data2
    train_data = mydataset(train_data, transform=transform, loader=Myloader)
    Dtr = DataLoader(dataset=train_data, batch_size=10, shuffle=True, num_workers=0)
    test_data = mydataset(test_data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=0)
    return train_data


# def get_sklearn_digits_dataset(loader):
#     """Downloads Sklearn handwritten digits dataset.
#     Uses the last SKLEARN_DIGITS_TEST_SIZE examples as the test
#     This is (hard-coded) -- do not change.
#
#     Parameters
#     ----------
#     loader : str (values: 'train' or 'test')."""
#     from torch.utils.data import Dataset
#     from sklearn.datasets import load_digits
#
#     class TorchDataset(Dataset):
#         """Abstracts a numpy array as a PyTorch dataset."""
#
#         def __init__(self, data, targets, transform=None):
#             self.data = torch.from_numpy(data).float()
#             self.targets = torch.from_numpy(targets).long()
#             self.transform = transform
#
#         def __getitem__(self, index):
#             x = self.data[index]
#             y = self.targets[index]
#             if self.transform:
#                 x = self.transform(x)
#             return x, y
#
#         def __len__(self):
#             return len(self.data)
#
#     transform = transforms.Compose(
#         [
#             transforms.ToPILImage(),
#             transforms.Resize(28),
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,)),
#         ]
#     )
#     # Get sklearn digits dataset
#     X_all, y_all = load_digits(return_X_y=True)
#     X_all = X_all.reshape((len(X_all), 8, 8))
#     y_train = y_all[:-SKLEARN_DIGITS_TEST_SIZE]
#     y_test = y_all[-SKLEARN_DIGITS_TEST_SIZE:]
#     X_train = X_all[:-SKLEARN_DIGITS_TEST_SIZE]
#     X_test = X_all[-SKLEARN_DIGITS_TEST_SIZE:]
#     if loader == "train":
#         return TorchDataset(X_train, y_train, transform=transform)
#     elif loader == "test":
#         return TorchDataset(X_test, y_test, transform=transform)
#     else:  # prama: no cover
#         raise ValueError("loader must be either str 'train' or str 'test'.")


class SimpleNet(nn.Module):
    """Basic Pytorch CNN for MNIST-like data."""

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        out = F.log_softmax(out,dim=1)
        return out


class CNN(BaseEstimator):
    def __init__(
            self,
            batch_size=64,
            epochs=6,
            log_interval=50,  # Set to None to not print
            lr=0.01,
            momentum=0.5,
            no_cuda=False,
            seed=1,
            test_batch_size=None,
            dataset="mnist",
            loader=None,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval = log_interval
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.seed = seed
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.seed)
        if self.cuda:  # pragma: no cover
            torch.cuda.manual_seed(self.seed)

        # Instantiate PyTorch model
        self.model = SimpleNet()
        if self.cuda:  # pragma: no cover
            self.model.cuda()

        self.loader_kwargs = {"num_workers": 1, "pin_memory": True} if self.cuda else {}
        self.loader = loader
        self._set_dataset(dataset)
        if test_batch_size is not None:
            self.test_batch_size = test_batch_size
        else:
            self.test_batch_size = self.test_size

    def _set_dataset(self, dataset):
        self.dataset = dataset
        if dataset == "mnist":
            # pragma: no cover
            self.get_dataset = load_data()
            self.train_size = MNIST_TRAIN_SIZE
            self.test_size = MNIST_TEST_SIZE
        # elif dataset == "sklearn-digits":
        #     self.get_dataset = get_sklearn_digits_dataset
        #     self.train_size = SKLEARN_DIGITS_TRAIN_SIZE
        #     self.test_size = SKLEARN_DIGITS_TEST_SIZE
        else:  # pragma: no cover
            raise ValueError("dataset must be 'mnist' or 'sklearn-digits'.")

    def get_params(self, deep=True):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "log_interval": self.log_interval,
            "lr": self.lr,
            "momentum": self.momentum,
            "no_cuda": self.no_cuda,
            "test_batch_size": self.test_batch_size,
            "dataset": self.dataset,
        }

    def set_params(self, **parameters):  # pragma: no cover
        for parameter, value in parameters.items():
            if parameter != "dataset":
                setattr(self, parameter, value)
        if "dataset" in parameters:
            self._set_dataset(parameters["dataset"])
        return self

    def fit(self, train_idx, train_labels=None, sample_weight=None, loader="train"):
        if self.loader is not None:
            loader = self.loader
        if train_labels is not None and len(train_idx) != len(train_labels):
            raise ValueError("Check that train_idx and train_labels are the same length.")

        if sample_weight is not None:  # pragma: no cover
            if len(sample_weight) != len(train_labels):
                raise ValueError(
                    "Check that train_labels and sample_weight " "are the same length."
                )
            class_weight = sample_weight[np.unique(train_labels, return_index=True)[1]]
            class_weight = torch.from_numpy(class_weight).float()
            if self.cuda:
                class_weight = class_weight.cuda()
        else:
            class_weight = None

        train_dataset = self.get_dataset

        # Use provided labels if not None o.w. use MNIST dataset training labels
        if train_labels is not None:
            sparse_labels = (
                    np.zeros(self.train_size if loader == "train" else self.test_size, dtype=int) - 1
            )
            sparse_labels[train_idx] = train_labels
            train_dataset.targets = sparse_labels

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=SubsetRandomSampler(train_idx),
            batch_size=self.batch_size,
            **self.loader_kwargs
        )

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Train for self.epochs epochs
        for epoch in range(1, self.epochs + 1):
            # Enable dropout and batch norm layers
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:  # pragma: no cover
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    target = np.array(target)
                    target = target.astype(float)
                    target = torch.from_numpy(target).long()
                    data=data.to(device)
                    target=target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target, class_weight)
                loss.backward()
                optimizer.step()
                if self.log_interval is not None and batch_idx % self.log_interval == 0:
                    print(
                        "TrainEpoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_idx),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        ),
                    )

    def predict(self, idx=None, loader=None):
        probs = self.predict_proba(idx, loader)
        return probs.argmax(axis=1)

    def predict_proba(self, idx=None, loader=None):
        if self.loader is not None:
            loader = self.loader
        if loader is None:
            is_test_idx = (
                    idx is not None
                    and len(idx) == self.test_size
                    and np.all(np.array(idx) == np.arange(self.test_size))
            )
            loader = "test" if is_test_idx else "train"
        dataset = self.get_dataset
        if idx is not None:
            if (loader == "train" and len(idx) != self.train_size) or (
                    loader == "test" and len(idx) != self.test_size
            ):
                dataset.data = np.array(dataset.data)[idx]
                dataset.targets = dataset.targets[idx]

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size if loader == "train" else self.test_batch_size,
            **self.loader_kwargs
        )
        self.model.eval()
        outputs = []
        for data, _ in loader:
            if self.cuda:  # pragma: no cover
                data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                output = self.model(data)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        # Convert to probabilities and return the numpy array of shape N x K
        out = outputs.cpu().numpy() if self.cuda else outputs.numpy()
        pred = np.exp(out)
        return pred
