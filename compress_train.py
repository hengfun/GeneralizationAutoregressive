from utils import CompressData
from models import Model
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class args(object):
    def __init__(self):
        self.batch_size = 512*32*4
        self.seq_length = 10
        self.input_size = 1
        self.dim = 1
        self.p_bias = 0.5
        self.stop_limit = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0 
        self.epochs = 10000
        self.hidden_size = 10
        self.layers = 2
        self.model_type = 'Seq2seq' #Options ["LSTM","RNN","Seq2seq","Transfomer"]
        self.optim = 'adam' # ['sgd', 'adam']
        self.learning_rate = 1e-3
        self.loss = "BCE" #Options ["MSE","BCE"]
        self.print_freq = 50
        self.save_dir = "logs"
        #if not os.path.exists('./logs'):
            #os.mkdir("/logs")
        

params = args()

if params.loss=="MSE":
    loss_fn = torch.nn.MSELoss()
else:
    loss_fn = torch.nn.BCEWithLogitsLoss()

sigmoid = torch.nn.Sigmoid()

stop_count = 0
prev_loss = 100

for p_bias in [.5,.8]:
    for seq_length in reversed(range(5,41,5)):
        for hidden_size in reversed([1,5,10,30]):
            params.seq_length =seq_length
            params.hidden_size = max(hidden_size,1)
            params.p_bias = p_bias

            model = Model(params)
            model = model.to(params.device)
            if params.optim == 'sgd':
                optim = torch.optim.SGD(model.parameters(), params.learning_rate)
            if params.optim == 'adam':
                optim = torch.optim.Adam(model.parameters(), params.learning_rate)

            coin_dataset = CompressData(params.p_bias, params.seq_length, params.epochs*params.batch_size)
            dataloader = DataLoader(coin_dataset, batch_size=params.batch_size,
                                    shuffle=False, num_workers=12)
            data ={i:0 for i in range(0,params.epochs)}
            for epoch, X in enumerate(dataloader):
                if epoch > params.epochs:
                    break
                X = X.to(params.device)
                
                X_hat = model(X.float())
                loss = loss_fn(X_hat, X.float())
                acc = ((sigmoid(X_hat) > 0.5) == X).float().mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
                data[epoch] = [loss.item(),acc.item()]

                if epoch % params.print_freq == 0:
                    # acc = (sigmoid(X_hat) > 0.5) == X
                    print('Prob {} | Seq_len {}| Hidden {} | Step {} | {} loss {:1.7f} | Acc {:1.6f}'.format(p_bias,seq_length,hidden_size,epoch, params.loss, loss.item(), acc.item()))
            pd.DataFrame(data).to_pickle('p{}seqlen{}hidden{}.pickle'.format(int(p_bias*100),seq_length,hidden_size))


    


