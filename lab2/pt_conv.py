import torch
import torchvision
from tqdm import tqdm
from torch import nn
import numpy as np
from torchvision.datasets import MNIST
import time
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'

class CovolutionalModel(nn.Module):
  def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
    super(CovolutionalModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding='same', bias=True)
    self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding='same', bias=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(7*7*32, fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.maxpool(h)
    h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
    h = self.conv2(h)
    h = self.maxpool(h)
    h = torch.relu(h)
    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    return torch.softmax(logits, dim=1)

def train(model, train_dataloader, valid_dataloader, config):
  writer = SummaryWriter(filename_suffix='conv')
  model.train()
  max_epochs = config['max_epochs']
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-2)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

  for epoch in range(1, max_epochs+1):
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Training (epoch={epoch}/{max_epochs})') as epoch_progress:
      for i, (X, Yoh_) in epoch_progress:
        Y_ = model(X)
        loss = nn.functional.cross_entropy(Y_, Yoh_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_progress.set_postfix(loss=loss.item())
        writer.add_scalar('train_loss', loss.item(), epoch * len(train_dataloader) + i)
    scheduler.step()
    model.eval()
    with torch.no_grad():
      for i, (X, Yoh_) in enumerate(valid_dataloader):
        Y_ = model(X)
        loss = nn.functional.cross_entropy(Y_, Yoh_)
        writer.add_scalar('valid_loss', loss.item(), epoch * len(valid_dataloader) + i)
    model.train()    
    writer.add_images('conv1', model.conv1.weight.view(-1, 1, 5, 5), epoch)
    writer.add_images('conv2', model.conv2.weight.view(-1, 1, 5, 5), epoch)
  writer.close()

if __name__=="__main__":
  #np.random.seed(100) 
  np.random.seed(int(time.time() * 1e6) % 2**31)

  ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True, transform=torchvision.transforms.ToTensor()), MNIST(DATA_DIR, train=False, transform=torchvision.transforms.ToTensor())
  n = len(ds_train)
  ds_train, mnist_val = torch.utils.data.random_split(ds_train, [int(n * 0.8), n - int(n * 0.8)])

  x_train, y_train = ds_train.dataset.data, ds_train.dataset.targets
  x_test, y_test = ds_test.data, ds_test.targets
  x_val, y_val = mnist_val.dataset.data, mnist_val.dataset.targets
  x_train, x_test, x_val = x_train.float().div_(255.0), x_test.float().div_(255.0), x_val.float().div_(255.0)

  C = y_train.max().item() + 1

  train_data = torch.utils.data.TensorDataset(x_train.view(-1, 1, 28, 28), y_train)
  val_data = torch.utils.data.TensorDataset(x_val.view(-1, 1, 28, 28), y_val)
  train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True)
  valid_dataloader = DataLoader(val_data, batch_size=50, shuffle=False)

  model = CovolutionalModel(1, 16, 32, 512, 10)
  config = {'max_epochs': 8}
  train(model, train_dataloader, valid_dataloader, config)
  model.eval()

  with torch.no_grad():
    X = x_test.view(-1, 1, 28, 28)
    Yoh_ = torch.eye(10)[y_test]
    Y_ = model(X)
    loss = nn.functional.cross_entropy(Y_, Yoh_)
    acc = (torch.argmax(Y_, dim=1) == y_test).float().mean()
    print(f'Test loss {loss.item()}')
    print(f'Test accuracy: {acc.item()}')
  torch.save(model.state_dict(), 'out/model.pth')

  