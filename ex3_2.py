import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from __future__ import print_function
import numpy as np
import random
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Network params
output_dim = 2
num_layers = 1
hidden_dim = 10
num_epochs = 10
BATCH_SIZE = 3
MAX_LENGTH = 6
input_size = 1


class LSTM(nn.Module):
 
    def __init__(self, input_size, hidden_size, batch_size, output_size=2,
                    num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
 
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
 
        # Define the output layer
        self.linear = nn.Linear(self.hidden_size, output_size)
 
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
 
    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, self.batch_size, self.hidden_size)
        
        c_0 = torch.zeros(
            self.num_layers, self.batch_size, self.hidden_size)
        
        # Propagate input through LSTM
        lstm_out, (hidden_out,_) = self.lstm(x, (h_0, c_0))
        
        lstm_out = lstm_out.view(self.hidden_size,-1)
        
        out = self.fc(lstm_out)
input_dict = {'00':0,'01':1,'10':2,'11':3}
output_dict_to_idx = {0:'0',1:'1'}
output_dict = {'0':0,'1':1}

def generateData(maxLength=MAX_LENGTH,batch=BATCH_SIZE):
    dec = np.random.randint(low = 0, high = 2**(maxLength-1), size=(batch,2)) 
    data = list()
    for item in dec:
        x = bin(item[0])[2:].zfill(maxLength)
        y = bin(item[1])[2:].zfill(maxLength)
        s = item[0] + item[1]
        sb = bin(s)[2:].zfill(maxLength)
        data.append([list(x),list(y),list(sb)])
    dataX_prepared = list()
    dataY_prepared = list()
    for item in data:
        x_single_batch = list()
        y_single_batch = list()
        x = item[0]
        y = item[1]
        o = item[2]
        it = zip(x,y,o)
        for i in it:
            temp = i[0] + i[1]
            x = input_dict[temp]
            y = output_dict[i[2]]
            x_single_batch.append([x])
            y_single_batch.append([y])
        dataX_prepared.append(x_single_batch)
        dataY_prepared.append(y_single_batch)
    return torch.FloatTensor(dataX_prepared),torch.FloatTensor(dataY_prepared)

#def prepareData(data):
#    for item in data:

dataX,Y = generateData()
model = LSTM(input_size=dataX.shape[2],hidden_size=hidden_dim,batch_size=dataX.shape[0])
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
def train(model,trainX,Y):
# Train the model
    for epoch in range(num_epochs):
        outputs = model(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, Y)
        
        loss.backward()
        
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            lossVector.append(loss.item())

    return lstm, lossVector
train(model, dataX,Y)

