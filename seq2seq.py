import torch
import torch.nn as nn



class Seq2seq(nn.Module):
    def __init__(self,input_size, hidden_size,layers):
        super(Seq2seq,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.build_model()
    def build_model(self):
        self.encoder = nn.LSTM(self.input_size,self.hidden_size,self.layers)
        self.decoder = nn.LSTM(self.input_size,self.hidden_size,self.layers)
    def forward(self,inputs):
        enc_out, enc_state = self.encoder(inputs)
        dec_out, dec_state = self.decoder(torch.zeros_like(inputs),enc_state)
        return dec_out,dec_state
