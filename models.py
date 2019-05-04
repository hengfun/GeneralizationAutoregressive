import torch
import torch.nn as nn
import numpy as np
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
        self.batch_size = params.batch_size
        
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
            # self.pos = torch.from_numpy(np.tile([i+1 for i in range(0,self.seq_length)],self.batch_size).reshape(-1,self.seq_length)).long().to(self.device)
            self.pos = (torch.arange(self.seq_length)+1).view(-1,1).to(self.device)#.repeat(1,)
            self.model = Transformer(self.input_size,self.input_size,self.seq_length,
                                    d_word_vec=self.hidden_size,d_model=self.hidden_size,
                                    d_inner=self.hidden_size*4,n_layers=1,
                                    tgt_emb_prj_weight_sharing=False,
                                    emb_src_tgt_weight_sharing=False)
            
    def forward(self,inputs):
        if self.model_type=="Transformer":
            logits = self.model(inputs,self.pos,inputs,self.pos)
            # logits = hidden
        else:
            hidden, _ = self.model(inputs)
            logits = self.dense(hidden)
        return logits
    