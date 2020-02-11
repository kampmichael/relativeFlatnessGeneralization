import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import torch.nn.init as init

BS = 32
LR = 0.0001
# layer of interest
layer_id = 8
dim = 84
# convergence criterion
epsilon = 1.0e-5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## get data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=False, num_workers=1)
testset = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

for exp in range(10):
    torch.manual_seed(exp)
    torch.cuda.manual_seed_all(exp)
    np.random.seed(exp)
    random.seed(exp)
    torch.backends.cudnn.deterministic = True

    # LeNet5
    class LeNet5(nn.Module):
        def __init__(self):
            super(LeNet5, self).__init__()

            self.conv1 = nn.Conv2d(3, 6, 5)
            init.xavier_normal_(self.conv1.weight.data)
            init.zeros_(self.conv1.bias.data)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            init.xavier_normal_(self.conv2.weight.data)
            init.zeros_(self.conv2.bias.data)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            init.xavier_normal_(self.fc1.weight.data)
            init.zeros_(self.fc1.bias.data)
            self.fc2 = nn.Linear(120, 84)
            init.xavier_normal_(self.fc2.weight.data)
            init.zeros_(self.fc2.bias.data)
            self.fc3 = nn.Linear(84, 10)
            init.xavier_normal_(self.fc3.weight.data)
            init.zeros_(self.fc3.bias.data)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = LeNet5()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    epoch = 0
    train = True
    while train:
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        grads = None
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # collect gradients from the layer for further flatness calculation
            i = 0
            for p in net.parameters():
                if i == layer_id:
                    if grads is None:
                        grads = p.grad.data.cpu().numpy()
                    else:
                        grads += p.grad.data.cpu().numpy()
                i += 1

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # check if all the gradients are less than epsilon for the stopping criterion
        all_grads_small = True
        for g in grads.flatten():
            all_grads_small = all_grads_small and (abs(g/(batch_idx+1)) <= epsilon)
        if all_grads_small:
            train = False

        print('Loss: %.10f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        epoch += 1

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Loss: %.10f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    torch.save(net.state_dict(), "cifar10_lenet5_bs" + str(BS) + "_lr" + str(LR) + "_exp" + str(exp))
