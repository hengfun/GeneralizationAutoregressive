from utils import CompressData
from models import Model
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class args(object):
    def __init__(self):
        self.batch_size = 50
        self.seq_length = 20
        self.input_size = 1
        self.dim = 1
        self.p_bias = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0 
        self.epochs = 500
        self.hidden_size = 10
        self.layers = 2
        self.model_type = 'Seq2seq' #Options ["LSTM","RNN","Seq2seq","Transfomer"]
        self.optim = 'adam' # ['sgd', 'adam']
        self.learning_rate = 1e-2
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


model = Model(params)
if params.optim == 'sgd':
    optim = torch.optim.SGD(model.parameters(), params.learning_rate)
if params.optim == 'adam':
    optim = torch.optim.Adam(model.parameters(), params.learning_rate)

coin_dataset = CompressData(params.p_bias, params.seq_length)
dataloader = DataLoader(coin_dataset, batch_size=params.batch_size,
                        shuffle=False, num_workers=1)

for epoch, X in enumerate(dataloader):
    if epoch > params.epochs:
        break
    X_hat = model(X.float())
    loss = loss_fn(X_hat, X.float())
    acc = (X_hat > 0.5) == X
    optim.zero_grad()
    loss.backward()
    optim.step()
    if epoch % params.print_freq == 0:
        print('Step {0} | {1} loss {2} | Acc {3}'.format(epoch, params.loss, loss.item(), acc.float().mean().item()))


    


