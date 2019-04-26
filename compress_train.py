from utils import CompressData
from models import Model
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class args(object):
    def __init__(self):
        self.batch_size = 256
        self.seq_length = 5
        self.input_size = 1
        self.dim = 1
        self.p_bias = 0.5
        self.stop_limit = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0 
        self.num_seeds =2
        self.epochs = 100000
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

prev_hidden_size = 1
new_hidden_size = 1

solved=False
done = False
while not done:
    params.hidden_size = new_hidden_size

    for seed in range(0,params.num_seeds):
        stop_count = 0
        prev_loss = 100

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
            acc_is_100 = int(acc.item())==1
            if prev_loss - loss.item()  < 1e-4 or acc_is_100:
                stop_count += 1
                if stop_count > params.stop_limit:
                    print('Converged, stoping after {0} steps'.format(epoch))
                    
                    print('last Acc {0}'.format(acc.item()))
                    solved =True
                    break
            else:
                stop_count = stop_count // 2

            if epoch % params.print_freq == 0:
                # acc = (sigmoid(X_hat) > 0.5) == X
                print('Seed {} | Seq_len {}| Hidden {} | Step {} | {} loss {:1.7f} | Acc {:1.6f}'.format(seed,params.seq_length,new_hidden_size,epoch, params.loss, loss.item(), acc.item()))
        
        #save logs
        # pd.DataFrame(data).to_pickle('Seed{}Len{}Hidden{}.pickle'.format(seed,seq_length,hidden_size))

    if not solved:
        print('Not solved, double hidden size')
        prev_hidden_size = new_hidden_size
        new_hidden_size= prev_hidden_size*2
        print('New Hidden size {}, Prev Hidden size {}'.format(new_hidden_size,prev_hidden_size))
    else:
        #solved
        print('Solved, Reduce hidden size')
        current_hidden_size = new_hidden_size
        new_hidden_size = int((current_hidden_size+prev_hidden_size)/2)
        prev_hidden_size = current_hidden_size
        print('New Hidden size {}, Prev Hidden size {}'.format(new_hidden_size,prev_hidden_size))
        diff = abs(prev_hidden_size-new_hidden_size)
        print('Abs(Prev Hidden - New Hidden) = Diff {}'.format(diff))
        if diff<1:
            print('Done, diff of {} <1'.format(diff))
            done=True
        

    


