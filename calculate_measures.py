import numpy as np
import torch.nn.functional as F
import os
import torch
from torch.autograd import grad
from torch import nn
import torchvision
import torchvision.transforms as transforms
import random
import torch.nn.init as init
from numpy import linalg as LA
import math

# layer for calculations - last hidden layer
layer_id = 8
reparam_layers = [6,7]
# dimensionality of the features
dim = 84

## get data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=1)
testset = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
inputs, labels = iter(trainloader).next()
x_train = inputs.cuda() # 1, 3, 32, 32,
y_train = labels.cuda() # 1,
inputs, labels = iter(testloader).next()
x_test = inputs.cuda() # 1, 3, 32, 32,
y_test = labels.cuda() # 1,
print("Data loaded")


# LeNet5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def last_layer(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x.view(-1, 84, 1)

    def classify(self, x):
        x = self.fc3(x)
        return x


# directory with saved network state dictionaries after training
for f in os.listdir("."):
    print(f, "----------------------")
    np.random.seed(None)
    # factor for reparametrization
    factor = np.random.randint(low=5, high=26) * 1.0
    print("Random reparametrization factor is", factor)

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True

    model = LeNet5().cuda()
    model.load_state_dict(torch.load(f))
    model.eval()

    loss = nn.CrossEntropyLoss(reduction='none')
    
    ## test loss
    test_output = model(x_test)
    test_loss = loss(test_output, y_test)
    # since size of the test dataset is 5 times smaller we multiply loss by 5
    test_loss = np.sum(test_loss.data.cpu().numpy()) * 5
    print("Test loss", test_loss)

    test_labels_np = y_test.data.cpu().numpy()
    test_output_np = test_output.data.cpu().numpy()
    test_acc = 0
    for i in range(len(test_labels_np)):
        if test_labels_np[i] == test_output_np[i].argmax():
            test_acc += 1
    print("Test accuracy is", test_acc*1.0/len(test_labels_np))

    ## train loss
    train_output = model(x_train)
    train_loss = loss(train_output, y_train)
    train_loss_overall = np.sum(train_loss.data.cpu().numpy())
    print("Train loss", train_loss_overall)

    train_labels_np = y_train.data.cpu().numpy()
    train_output_np = train_output.data.cpu().numpy()
    train_acc = 0
    for i in range(len(train_labels_np)):
        if train_labels_np[i] == train_output_np[i].argmax():
            train_acc += 1
    print("Train accuracy is", train_acc*1.0/len(train_labels_np))

    ## calculate FisherRao norm
    # analytical formula for crossentropy loss from Appendix of the original paper
    sum_derivatives = 0
    m = nn.Softmax(dim=0)
    for inp in range(len(train_output)):
        sum_derivatives += \
            (np.inner(m(train_output[inp]).data.cpu().numpy(), train_output[inp].data.cpu().numpy()) -
             train_output[inp].data.cpu().numpy()[train_labels_np[inp]]) ** 2
    fr_norm_origin = math.sqrt(((5 + 1) ** 2) * (1.0 / len(train_output)) * sum_derivatives)
    print("Fisher Rao norm is", fr_norm_origin)

    loss = nn.CrossEntropyLoss()
    train_loss = loss(train_output, y_train)
    print("Train loss is", train_loss)

    # hessian calculation for the layer of interest
    i = 0
    for p in model.parameters():
        if i == layer_id:
            last_layer = p
        i += 1
    last_layer_jacobian = grad(train_loss, last_layer, create_graph=True, retain_graph=True)
    hessian = []
    for n_grd in last_layer_jacobian[0]:
        for w_grd in n_grd:
            drv2 = grad(w_grd, last_layer, retain_graph=True)
            hessian.append(drv2[0].data.cpu().numpy().flatten())

    sum = 0.0
    for n in last_layer.data.cpu().numpy():
        for w in n:
            sum += w**2
    print("squared euclidian norm is calculated", sum)

    max_eignv = LA.eigvalsh(hessian)[-1]
    print("largest eigenvalue is", max_eignv)

    trace = np.trace(hessian)
    norm_trace = trace / (1.0*len(hessian))
    print("normalized trace is", norm_trace)

    print("eigenvalue relative flatness is ", sum * max_eignv)
    print("tracial flatness is ", sum * norm_trace)

    # apply simplest reparametrization for ReLU network
    model = LeNet5().cuda()
    model.load_state_dict(torch.load(f))
    i = 0
    for l in model.parameters():
        if i in reparam_layers:
            l.data = l.data * 1.0/ factor
        elif i == layer_id:
            l.data = l.data * factor
        i += 1
    model.eval()

    loss = nn.CrossEntropyLoss(reduction='none')
    
    ## test loss
    test_output = model(x_test)
    test_loss = loss(test_output, y_test)
    # since size of the test dataset is 5 times smaller we multiply loss by 5
    test_loss = np.sum(test_loss.data.cpu().numpy()) * 5
    print("Test loss", test_loss)

    test_labels_np = y_test.data.cpu().numpy()
    test_output_np = test_output.data.cpu().numpy()
    test_acc = 0
    for i in range(len(test_labels_np)):
        if test_labels_np[i] == test_output_np[i].argmax():
            test_acc += 1
    print("Test accuracy is", test_acc*1.0/len(test_labels_np))

    ## train loss
    train_output = model(x_train)
    train_loss = loss(train_output, y_train)
    train_loss_overall = np.sum(train_loss.data.cpu().numpy())
    print("Train loss", train_loss_overall)

    train_labels_np = y_train.data.cpu().numpy()
    train_output_np = train_output.data.cpu().numpy()
    train_acc = 0
    for i in range(len(train_labels_np)):
        if train_labels_np[i] == train_output_np[i].argmax():
            train_acc += 1
    print("Train accuracy is", train_acc*1.0/len(train_labels_np))

    ## calculate FisherRao norm
    # analytical formula for crossentropy loss from the original paper
    train_labels_np = y_train.data.cpu().numpy()
    sum_derivatives = 0
    for inp in range(len(train_output)):
        sum_derivatives += \
            (np.inner(m(train_output[inp]).data.cpu().numpy(), train_output[inp].data.cpu().numpy()) -
             train_output[inp].data.cpu().numpy()[train_labels_np[inp]]) ** 2
    fr_norm = math.sqrt(((5 + 1) ** 2) * (1.0 / len(train_output)) * sum_derivatives)
    print("Fisher Rao norm is", fr_norm)

    loss = nn.CrossEntropyLoss()
    train_loss = loss(train_output, y_train)
    print("Train loss is", train_loss)

    i = 0
    last_layer_id = 8
    for p in model.parameters():
        if i == last_layer_id:
            last_layer = p
        i += 1
    last_layer_jacobian = grad(train_loss, last_layer, create_graph=True, retain_graph=True)
    hessian = []
    for n_grd in last_layer_jacobian[0]:
        for w_grd in n_grd:
            drv2 = grad(w_grd, last_layer, retain_graph=True)
            hessian.append(drv2[0].data.cpu().numpy().flatten())

    sum = 0.0
    for n in last_layer.data.cpu().numpy():
        for w in n:
            sum += w**2
    print("squared euclidian norm is calculated", sum)

    max_eignv = LA.eigvalsh(hessian)[-1]
    print("largest eigenvalue is", max_eignv)

    trace = np.trace(hessian)
    norm_trace = trace / (1.0*len(hessian))
    print("normalized trace is", norm_trace)

    print("eigenvalue relative flatness is ", sum * max_eignv)
    print("tracial flatness is ", sum * norm_trace)

    ### feature robustness calculation
    a = np.zeros((dim, dim, dim))
    np.fill_diagonal(a, 1)
    delta = 0.001
    robustness = []
    loss = nn.CrossEntropyLoss(reduction='none')
    train_loss_overall = loss(train_output, y_train)
    train_loss_overall = np.sum(train_loss_overall.data.cpu().numpy())

    ## train loss with noisy features
    for k in range(dim):
        activation = model.last_layer(x_train).data.cpu().numpy()
        added_noise = []
        for i in range(50000):
            added_noise.append(activation[i] + delta * np.dot(a[k], activation[i]))
        added_noise = np.array(added_noise).reshape(-1, dim)
        del activation
        train_output = model.classify(torch.cuda.FloatTensor(added_noise))
        train_loss_with_noise = loss(train_output, y_train)
        train_loss_overall_with_noise = np.sum(train_loss_with_noise.data.cpu().numpy())
        robustness.append((1.0 / len(train_loss_with_noise)) * (train_loss_overall_with_noise - train_loss_overall))
    print("Robustness", robustness)
