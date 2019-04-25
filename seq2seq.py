import torch
import torch.nn as nn



class Seq2seq(nn.Module):
    def __init__(self,input_size, hidden_size,layers, rnn='LSTM'):
        super(Seq2seq,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.build_model(rnn)
    def build_model(self,rnn):
        if rnn == 'LSTM':
            self.encoder = nn.LSTM(self.input_size,self.hidden_size,self.layers, batch_first=True)
            self.decoder = nn.LSTM(self.input_size,self.hidden_size,self.layers, batch_first=True)
        if rnn == 'GRU':
            self.encoder = nn.GRU(self.input_size,self.hidden_size,self.layers, batch_first=True)
            self.decoder = nn.GRU(self.input_size,self.hidden_size,self.layers, batch_first=True)

    def forward(self,inputs):
        enc_out, enc_state = self.encoder(inputs)
        dec_out, dec_state = self.decoder(torch.zeros_like(inputs),enc_state)
        return dec_out, dec_state
