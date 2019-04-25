from utils import CompressData
from models import Model
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class args(object):
    def __init__(self):
        self.batch_size = 80
        self.seq_length = 10
        self.input_size = 1
        self.dim = 1
        self.p_bias = 0.5
        self.stop_limit = 50
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0 
        self.epochs = 11000
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



stop_count = 0
prev_loss = 100

model = Model(params)
if params.optim == 'sgd':
    optim = torch.optim.SGD(model.parameters(), params.learning_rate)
if params.optim == 'adam':
    optim = torch.optim.Adam(model.parameters(), params.learning_rate)

coin_dataset = CompressData(params.p_bias, params.seq_length, params.epochs*params.batch_size)
dataloader = DataLoader(coin_dataset, batch_size=params.batch_size,
                        shuffle=False, num_workers=1)

sigmoid = torch.nn.Sigmoid()
for epoch, X in enumerate(dataloader):
    if epoch > params.epochs:
        break
    X_hat = model(X.float())
    loss = loss_fn(X_hat, X.float())
    #acc = (sigmoid(X_hat) > 0.5) == X
    optim.zero_grad()
    loss.backward()
    optim.step()
    if prev_loss - loss.item()  < 1e-4:
        stop_count += 1
        if stop_count > params.stop_limit:
            print('Converged, stoping after {0} steps'.format(epoch))
            acc = (sigmoid(X_hat) > 0.5) == X
            print('last Acc {0}'.format(acc.float().mean()))
            break
    else:
        stop_count = stop_count // 2
    if epoch % params.print_freq == 0:
        acc = (sigmoid(X_hat) > 0.5) == X
        print('Step {0} | {1} loss {2} | Acc {3}'.format(epoch, params.loss, loss.item(), acc.float().mean().item()))


    


