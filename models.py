import torch
import torch.nn as nn
from transformer.Models import Transformer
from seq2seq import Seq2seq


class Model(nn.Module):
    def __init__(self,params):
        super(Model,self).__init__()
        self.seq_length = params.seq_length
        self.hidden_size = params.hidden_size
        self.input_size = params.input_size
        self.model_type = params.model_type
        self.layers = params.layers
        self.loss = params.loss
        self.output_size = params.dim
        self.device = params.device
        self.build_model(rnn_type=params.rnn_type)

    def build_model(self, rnn_type=None):
        if self.model_type=="LSTM":
            self.model = nn.LSTM(self.input_size,self.hidden_size,self.layers)
            self.dense = nn.Linear(self.hidden_size,self.output_size)
        if self.model_type=='RNN':
            self.model = nn.RNN(self.input_size,self.hidden_size)
            self.dense = nn.Linear(self.hidden_size,self.output_size)
        if self.model_type == "Seq2seq":
            self.model = Seq2seq(self.input_size,self.hidden_size,self.layers, rnn=rnn_type)   
            self.dense = nn.Linear(self.hidden_size,self.output_size)
        if self.model_type=="Transformer":
            self.model = Transformer(self.input_size,self.input_size,self.seq_length)

    def forward(self,inputs):
        hidden, _ = self.model(inputs)
        logits = self.dense(hidden)
        return logits