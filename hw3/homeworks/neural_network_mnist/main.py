# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = 1 / math.sqrt(d)
        self.W_0 = torch.FloatTensor(d, h).uniform_(-alpha, alpha)
        self.W_0.requires_grad = True
        self.b_0 = torch.FloatTensor(1, h).uniform_(-alpha, alpha)
        self.b_0.requires_grad = True
        self.W_1 = torch.FloatTensor(h, k).uniform_(-alpha, alpha)
        self.W_1.requires_grad = True
        self.b_1 = torch.FloatTensor(1, k).uniform_(-alpha, alpha)
        self.b_1.requires_grad = True

        self.params = [self.W_0, self.b_0, self.W_1, self.b_1]
        #raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        x = torch.matmul(x, self.W_0) + self.b_0
        x = torch.matmul(relu(x), self.W_1) + self.b_1
        return x
        #raise NotImplementedError("Your Code Goes Here")


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = 1 / math.sqrt(d)
        self.W_0 = torch.FloatTensor(d, h0).uniform_(-alpha, alpha)
        self.W_0.requires_grad = True
        self.b_0 = torch.FloatTensor(1, h0).uniform_(-alpha, alpha)
        self.b_0.requires_grad = True
        self.W_1 = torch.FloatTensor(h1, h0).uniform_(-alpha, alpha)
        self.W_1.requires_grad = True
        self.b_1 = torch.FloatTensor(1, h1).uniform_(-alpha, alpha)
        self.b_1.requires_grad = True
        self.W_2 = torch.FloatTensor(h1, k).uniform_(-alpha, alpha)
        self.W_2.requires_grad = True
        self.b_2 = torch.FloatTensor(1, k).uniform_(-alpha, alpha)
        self.b_2.requires_grad = True

        self.params = [self.W_0, self.b_0, self.W_1, self.b_1, self.W_2, self.b_2]
        #raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        x = torch.matmul(x, self.W_0) + self.b_0
        x = torch.matmul(relu(x), self.W_1) + self.b_1
        x = torch.matmul(relu(x), self.W_2) + self.b_2
        return x
        #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    avg_losses = []
    accuracy = 0.0
    epoch = 0
    train_size = len(train_loader.dataset)
    while accuracy < 0.99:
        epoch_loss_count = 0
        acc = 0
        for (x,y) in tqdm(train_loader):
            x = x.view(-1, 784)
            optimizer.zero_grad()
            logits = model.forward(x)
            preds = torch.argmax(logits, 1)
            acc += torch.sum(preds == y).item()
            loss = cross_entropy(logits, y)
            epoch_loss_count += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss_count / train_size
        accuracy = acc / train_size
        avg_losses.append(epoch_loss)

        print("Epoch: ", epoch)
        print("Loss: ", epoch_loss)
        print("Accuracy: ", accuracy)
        
        epoch += 1
    return avg_losses
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset)

    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset)

    F1_model = F1(64, 784, 10)
    F1_optimizer = Adam(F1_model.params, lr=1e-4)
    F1_num_params = 0
    for param in F1_model.params:
        F1_num_params += torch.prod(torch.tensor(param.shape))
    print("F1_num_params=", F1_num_params)
    F1_losses = train(F1_model, F1_optimizer, dataloader)

    plt.figure(1)
    plt.plot(torch.arange(0, len(F1_losses), 1), F1_losses, "-o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("F1.png")

    
    F2_model = F2(32, 32, 784, 10)
    F2_optimizer = Adam(F2_model.params, lr=1e-4)
    F2_num_params = 0
    for param in F2_model.params:
        F2_num_params += torch.prod(torch.tensor(param.shape))
    print("F2_num_params=", F2_num_params)
    F2_losses = train(F2_model, F2_optimizer, dataloader)

    plt.figure(2)
    plt.plot(torch.arange(0, len(F2_losses), 1), F2_losses, "-o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("F2.png")
    

    for model in [F1_model,F2_model]:
        epoch_loss_count = 0
        acc = 0
        for (x,y) in test_dataloader:
            x = x.view(-1, 784)
            logits = model.forward(x)
            preds = torch.argmax(logits, 1)
            acc += torch.sum(preds == y).item()
            loss = cross_entropy(logits, y)
            epoch_loss_count += loss.item()

        print("Testing Results")
        print("Loss: ", epoch_loss_count / len(test_dataloader))
        print("Accuracy: ", acc / len(test_dataloader.dataset))


    #raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
