
#from __future__ import print_function
import numpy as np
from time import sleep
import random
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
random.seed( 10 )

"""## Preparing the input data ##

###   Radom binary strings of required length as training data ###
- The  function <i>getSample()</i> takes a string-length as input and then returns the input vector and target vector that need to be fed to the RNN
- Say if your string-length is 2, lower and upper bounds would be 2 and 3. 
- Then if the two random numbers picked from this range are 2 and 3 ( you have only 2 and 3 in that range :) )
- your inputs in binary would be 10 and 11 and your sum is 5 which is 101.
- <b> Padding :</b>Since your ouput is one bit longer we will rewrite the inputs too in 3 bit form so  010 + 011 -- > 101


### Training data as input sequene and target sequence pairs  ###
Starting from the least significant bit  (since the addition starts from LSB) we concatenate the correspodning bits in each input binary string and that forms our input sequence.
And your target vector would be the ourput binary string reversed (Since you start from LSB)

Hence  your input at one timestep is this ordered pair of bits for that particular position and target for that timestep would be the corresponding bit in the output string

so your input dimension at each time step is 2 and target dimesnion is 1

in the above case so your input and target pairs would be

[1 0] - > 1 <br>
[1 1]  -> 0 <br>
[0 0]  -> 0


![title](binaryinput.jpg)
"""

def getSample(stringLength, testFlag):
  #takes stringlength as input 
  #returns a sample for the network - an input sequence - x and its target -y
  #x is a T*2 array, T is the length of the string and 2 since we take one bit each from each string
  #testFlag if set prints the input numbers and its sum in both decimal and binary form
  lowerBound=pow(2,stringLength-1)+1
  upperBound=pow(2,stringLength)

  num1=random.randint(lowerBound,upperBound)
  num2=random.randint(lowerBound,upperBound)

  num3=num1+num2
  num3Binary=(bin(num3)[2:])

  num1Binary=(bin(num1)[2:])

  num2Binary=(bin(num2)[2:])

  if testFlag==1:
    print('input numbers and their sum  are', num1, ' ', num2, ' ', num3)
    print ('binary strings are', num1Binary, ' ' , num2Binary, ' ' , num3Binary)
  len_num1= (len(num1Binary))

  len_num2= (len(num2Binary))
  len_num3= (len(num3Binary))

  # since num3 will be the largest, we pad  other numbers with zeros to that num3_len
  num1Binary= ('0'*(len(num3Binary)-len(num1Binary))+num1Binary)
  num2Binary= ('0'*(len(num3Binary)-len(num2Binary))+num2Binary)


  # forming the input sequence
  # the input at first timestep is the least significant bits of the two input binary strings
  # x will be then a len_num3 ( or T ) * 2 array
  x=np.zeros((len_num3,2),dtype=np.float32)
  for i in range(0, len_num3):
    x[i,0]=num1Binary[len_num3-1-i] # note that MSB of the binray string should be the last input along the time axis
    x[i,1]=num2Binary[len_num3-1-i]
  # target vector is the sum in binary
  # convert binary string in <string> to a numpy 1D array
  #https://stackoverflow.com/questions/29091869/convert-bitstring-string-of-1-and-0s-to-numpy-array
  y=np.array(list(map(int, num3Binary[::-1])))
  #print (x)
  #print (y)
  return x,y

"""## How does the network look like ? ##
The figure below shows  fully rolled network for our task for the input - target pair we took as an example earlier.
In the figure, for ease of drawing, hiddenDIm is chosen as 2
![network architecture](binarynet.jpg)
"""

class Adder (nn.Module):
  def __init__(self, inputDim, hiddenDim, outputDim):
    super(Adder, self).__init__()
    self.inputDim=inputDim
    self.hiddenDim=hiddenDim
    self.outputDim=outputDim
    self.lstm=nn.RNN(inputDim, hiddenDim )
    self.outputLayer=nn.Linear(hiddenDim, outputDim)
    self.sigmoid=nn.Sigmoid()
  def forward(self, x ):
    #size of x is T x B x featDim
    #B=1 is dummy batch dimension added, because pytorch mandates it
    #if you want B as first dimension of x then specift batchFirst=True when LSTM is initalized
    #T,D  = x.size(0), x.size(1)
    #batch is a must 
    lstmOut,_ =self.lstm(x ) #x has two  dimensions  seqLen *batch* FeatDim=2
    T,B,D  = lstmOut.size(0),lstmOut.size(1) , lstmOut.size(2)
    lstmOut = lstmOut.contiguous() 
        # before  feeding to linear layer we squash one dimension
    lstmOut = lstmOut.view(B*T, D)
    outputLayerActivations=self.outputLayer(lstmOut)
    #reshape actiavtions to T*B*outputlayersize
    outputLayerActivations=outputLayerActivations.view(T,B,-1).squeeze(1)
    outputSigmoid=self.sigmoid(outputLayerActivations)
    return outputSigmoid

"""### traning the network ###

- batch learning is not used, only one seqeuence is fed at a time
- runs purely on a cpu
- MSE loss is used
"""

featDim=2 # two bits each from each of the String
outputDim=1 # one output node which would output a zero or 1

lstmSize=10

lossFunction = nn.MSELoss()
model =Adder(featDim, lstmSize, outputDim)
print ('model initialized')
#optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.8)
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=50
### epochs ##
totalLoss= float("inf")
while totalLoss > 1e-5:
  print(" Avg. Loss for last 500 samples = %lf"%(totalLoss))
  totalLoss=0
  for i in range(0,epochs): # average the loss over 200 samples
    
    stringLen=4
    testFlag=0
    x,y=getSample(stringLen, testFlag)

    model.zero_grad()


    x_var=torch.from_numpy(x).unsqueeze(1).float() #convert to torch tensor and variable
    # unsqueeze() is used to add the extra dimension since
    # your input need to be of t*batchsize*featDim; you cant do away with the batch in pytorch
    seqLen=x_var.size(0)
    #print (x_var)
    x_var= x_var.contiguous()
    #y_var=torch.from_numpy(y).float()
    y_var= torch.from_numpy(y)
    finalScores = model(x_var)
    #finalScores=finalScores.
    y_var = y_var.type(torch.FloatTensor)
    loss=lossFunction(finalScores,y_var)  
    totalLoss+=loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  
  totalLoss=totalLoss/epochs
  break

"""### Testing the model ###
Remember that the network was purely trained on strings of length =3 <br>
now lets the net on bitstrings of length=4
"""

stringLen=5
testFlag=1
# test the network on 10 random binary string addition cases where stringLen=4
for i in range (0,10):
	x,y=getSample(stringLen,testFlag)
	x_var=autograd.Variable(torch.from_numpy(x).unsqueeze(1).float())
	y_var=autograd.Variable(torch.from_numpy(y).float())
	seqLen=x_var.size(0)
	x_var= x_var.contiguous()
	finalScores = model(x_var).data.t()
	#print(finalScores)
	bits=finalScores.gt(0.5)
	bits=bits[0].numpy()

	print ('sum predicted by RNN is ',bits[::-1])
	print('##################################################')

"""### Things to try out 
- See that increasing the hidden size to say 100 worsens the performance
- Change the model slightly to use NLL loss or cross entropy loss (you may want to add two output nodes in this case; one for 1 and one for 0.)
"""

