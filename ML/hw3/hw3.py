import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as ts
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

def extTrainData():
  data = np.loadtxt("train.csv", dtype=np.str, delimiter=",", encoding='ISO-8859-1')
  upTo = 20001
  y = data[1:upTo,0].astype(float)
  x = np.zeros([y.shape[0], 48*48])
  for i in range(x.shape[0]):
    x[i,:] = np.array(data[i+1,1].split(' ')).astype(float)/255
  return x,y

def extTestData():
  data = np.loadtxt("test.csv", dtype=np.str, delimiter=",", encoding='ISO-8859-1')
  x = np.zeros([data.shape[0]-1, 48*48])
  for i in range(x.shape[0]):
    x[i,:] = np.array(data[i+1,1].split(' ')).astype(float)/255
  return x

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
    self.conv4 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
    self.conv5 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
    self.conv6 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
    self.mp = nn.MaxPool2d(2)
    self.lrelu = nn.LeakyReLU(0.2)
    self.bn64 = nn.BatchNorm2d(64)
    self.bn128 = nn.BatchNorm2d(128)
    self.bn256 = nn.BatchNorm2d(256)
    self.fc1 = nn.Linear(256*6*6 ,1024)
    self.fc2 = nn.Linear(1024 ,512)
    self.fc3 = nn.Linear(512, 7)
    self.dropout = nn.Dropout(0.5)
  
  def forward(self, x):
    x = self.lrelu(self.bn64(self.conv1(x)))
    x = self.lrelu(self.bn64(self.conv2(x)))
    x = self.mp(x)
    x = self.lrelu(self.bn128(self.conv3(x)))
    x = self.lrelu(self.bn128(self.conv4(x)))
    x = self.mp(x)
    x = self.lrelu(self.bn256(self.conv5(x)))
    x = self.lrelu(self.bn256(self.conv6(x)))
    x = self.mp(x)

    x = x.view(-1, 256*6*6)
    x = self.lrelu(self.fc1(x))
    x = self.dropout(x)
    x = self.lrelu(self.fc2(x))
    x = self.dropout(x)
    x = F.log_softmax(self.fc3(x))
    return x


if __name__=="__main__":
  BATCH_SIZE = 500

  x_train, y_labeled = extTrainData()
  x_train = torch.tensor(x_train).view(x_train.shape[0],1,48,48)
  y_labeled = torch.tensor(y_labeled).view(-1)
  dataset = Data.TensorDataset(x_train, y_labeled)
  train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
  net = Net()
  net = net.double()
  #optimizer = optim.RMSprop(net.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.Adam(net.parameters(), lr=0.01)
  iteration = 20
  y_train = torch.zeros(1)
  for i in range(iteration):
    for step, (batch_x, batch_y) in enumerate(train_loader):
      y_train = net(batch_x.double())
      criterion = nn.CrossEntropyLoss()
      loss = criterion(y_train, batch_y.long())
      optimizer.zero_grad()
      loss.backward()
      print(i+1, 'iteration', step+1, 'step')
      print(loss)
      optimizer.step()
  y_train = net(x_train.double())
  y_pred = torch.max(y_train, 1)[1] #.data.numpy().squeeze()
  print(y_pred[:20], y_labeled[:20].long())
  print((y_pred==y_labeled.long()).sum().item()/y_labeled.shape[0])

  x_test = extTestData()
  x_test = torch.tensor(x_test).view(x_test.shape[0],1,48,48)
  y_test = net(x_test)
  y_pred = torch.max(y_test, 1)[1].numpy()
  print(y_pred[:20])
  
  y = [["id","label"]]
  for i in range(y_pred.shape[0]):
    y.append(["{0}".format(i), "{0}".format(y_pred[i])])
  #print(w)
  #print(y)
  y = np.array(y)

  #save w and predict.csv
  np.savetxt("predict.csv", y, delimiter=",", fmt="%s")
