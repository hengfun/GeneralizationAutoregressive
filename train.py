from utils import Data
from models import Model
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

class args(object):
    def __init__(self):
        self.batch_size = 20
        self.num_samples = 100
        self.min_seq_length = 1
        self.max_seq_length = 20
        #self.num_batches = int(self.num_samples/self.batch_size/self.seq_length)
        self.input_size =1
        self.dim = 1
        self.p_bias = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_val_test_split = [1,0,0] #going to mix in some non-random sequences duh!
        self.seed = 0 
        self.epochs = 400-1
        self.hidden_size = 10
        self.layers = 2
        self.model_type = 'Seq2seq' #Options ["LSTM","RNN","Seq2seq","Transfomer"]
        self.optim = 'sgd'
        self.learning_rate = 1e-2
        self.random =True
        self.loss = "MSE" #Options ["MSE","BCE"]
        self.print_freq = 100
        self.save_dir = "logs"
        if not os.path.exists('./logs'):
            os.mkdir("/logs")
        

params = args()

#dimension [batches, seq_length,batch_size,dim]
#data = Data(params)
#train_batch_size,seq_length,batch_size,dim = data.random_xtrain.shape

#model = Model(params)

if params.loss=="MSE":
    loss_fn = torch.nn.MSELoss()
else:
    loss_fn = torch.nn.BCEWithLogitsLoss()


true_deviation = {}

for length in range(params.min_seq_length, params.max_seq_length+1):
    print('Seq Length: {0}'.format(length))
    params.seq_length = length
    model = Model(params)
    if params.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    elif params.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    data = Data(params,length)
    
    train_batch_size,seq_length,batch_size,dim = data.random_xtrain.shape
    for e in range(params.epochs + 1):
        epoch_loss =0
        deviation = []
        for batch in range(train_batch_size):
            if params.random:
                inputs = data.random_xtrain[batch] # random data of coin flips 
                y = data.random_ytrain[batch]
            #train pseudo if not random
            else:
                inputs = data.pseudo_xtrain[batch] # pseudo xtrain data [1,0,1,0]
                y = data.pseudo_xtrain[batch]
            
            y_predict= model(inputs) #output, [seq_length,batch_size,dim]
            if e < params.epochs:    
                loss = loss_fn(y_predict,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss +=loss.item() / train_batch_size
            else:
                if params.loss!='MSE':
                    predict = F.sigmoid(y_predict)
                else:
                    predict = y_predict
                deviation.append(torch.abs(0.5 - y_predict).mean().item())

        if e%params.print_freq==0:
            if params.loss!='MSE':
                predict = F.sigmoid(y_predict)
            else:
                predict = y_predict
            print("Epoch:{} Loss:{}".format(e,round(epoch_loss,5)))
            #print("Probabilities :{}".format(predict[:,0,:].flatten()))
        if e == params.epochs:
            true_deviation[length] = sum(deviation) / len(deviation)

import numpy as np
import matplotlib.pyplot as plt 
x_ = []
y_ = []
for k,v in true_deviation.items():
    x_.append(int(k))
    y_.append(float(v))
x_ = np.array(x_,dtype=np.int32)
y_ = np.array(y_,dtype=np.float32)

plt.plot(x_,y_)
plt.show()


        
