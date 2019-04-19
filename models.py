import torch
import torch.nn as nn
from transformer.Models import Transformer


class Model(nn.Module):
    def __init__(self,params):
        super(Model,self).__init__()
        self.hidden_size = params.hidden_size
        self.input_size = params.input_size
        self.model_type = params.model_type
        self.output_size = params.dim
        self.device = params.device
        self.build_model()

    def build_model(self):
        if self.model_type=="LSTM":
            self.model = nn.LSTM(self.input_size,self.hidden_size)
            self.dense = nn.Linear(self.hidden_size,self.output_size)
        if self.model_type=='RNN':
            self.model = nn.RNN(self.input_size,self.hidden_size)
            self.dense = nn.Linear(self.hidden_size,self.output_size)
        if self.model_type=="Transformer":
            self.model = Transformer()

    def forward(self,inputs):
        out, _ = self.model(inputs)
        output = self.dense(out)
        return output