from utils import data
from models import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class args(object):
    def __init__(self):
        self.batch_size = 250
        self.num_samples = 100000
        self.seq_length = 10
        self.num_batches = int(self.num_samples/self.batch_size/self.seq_length)
        self.input_size =1
        self.dim = 1
        self.device = device
        self.train_val_test_split = [.6,.2,.2] #going to mix in some non-random sequences duh!
        self.seed = 0 
        self.epochs = 1
        self.hidden_size = 10
        self.layers = 1
        self.model_type = 'LSTM'


params = args()

d = data(params)

model = Model(params)

for e in range(params.epochs):
    train_batch_size = d.r_xtrain.shape[0]
    for batch in range(train_batch_size):
        inputs = torch.from_numpy(d.r_xtrain[batch]).float()
        predict= model(inputs)
