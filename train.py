from utils import Data
from models import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class args(object):
    def __init__(self):
        self.batch_size = 100
        self.num_samples = 100000
        self.seq_length = 10
        self.num_batches = int(self.num_samples/self.batch_size/self.seq_length)
        self.input_size =1
        self.dim = 1
        self.device = device
        self.train_val_test_split = [.6,.2,.2] #going to mix in some non-random sequences duh!
        self.seed = 0 
        self.epochs = 100
        self.hidden_size = 20
        self.layers = 1
        self.model_type = 'LSTM'
        self.learning_rate = 1e-1
        self.random =True
        self.loss = "BCE"


params = args()

#dimension [batches, seq_length,batch_size,dim]
data = Data(params)
model = Model(params)

if params.loss=="MSE":
    loss_fn = torch.nn.MSELoss(reduction='mean')
else:
    loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
for e in range(params.epochs):
    train_batch_size = data.random_xtrain.shape[0]
    epoch_loss =0
    for batch in range(train_batch_size):
        
        if params.random:
            inputs = torch.from_numpy(data.random_xtrain[batch]).float() # random data of coin flips 
            y = torch.from_numpy(data.random_ytrain[batch]).float()
        #train psuedo if not random
        else:
            inputs = torch.from_numpy(data.pseudo_xtrain[batch]).float() # psuedo xtrain data [1,0,1,0]
            y = torch.from_numpy(data.pseudo_xtrain[batch]).float()

        
        y_predict= model(inputs) #output, [seq_length,batch_size,dim]
        
        
        loss = loss_fn(y_predict,y)
        epoch_loss +=loss
        
        predict = F.sigmoid(y_predict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if e%10==0:
        print("Epoch:{} Loss:{}".format(e,epoch_loss))
        print("Probabilities :{}".format(predict[:,0,:].flatten()))

        
