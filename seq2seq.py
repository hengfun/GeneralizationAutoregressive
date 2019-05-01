import torch
import torch.nn as nn



class Seq2seq(nn.Module):
    def __init__(self,input_size, hidden_size,layers, rnn='LSTM'):
        super(Seq2seq,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.initializer_hh = torch.nn.init.orthogonal_
        self.initializer_ih = torch.nn.init.xavier_normal_
        # self.forget_bias = torch.tensor(2.0,requires_grad=True).float()
        self.forget_bias =2.0
        self.build_model(rnn)
        
        

    def build_model(self,rnn):
        if rnn == 'LSTM':
            self.encoder = self.init_lstm()
            self.decoder = self.init_lstm()
        if rnn == 'GRU':
            self.encoder = nn.GRU(self.input_size,self.hidden_size,self.layers, batch_first=True)
            self.decoder = nn.GRU(self.input_size,self.hidden_size,self.layers, batch_first=True)

    def init_lstm(self):
        lstm = nn.LSTM(self.input_size,self.hidden_size,self.layers, batch_first=True)
        for name,weights in lstm.named_parameters():
            # print(name,weights)
            if "bias_hh" in name:
                #weights are initialized 
                #(b_hi|b_hf|b_hg|b_ho), 
                weights[self.hidden_size:self.hidden_size*2].data.fill_(self.forget_bias)
            elif 'bias_ih' in name:
                 #(b_ii|b_if|b_ig|b_io)
                pass
            elif "weight_hh" in name:
                self.initializer_hh(weights)
            elif 'weight_ih' in name:
                self.initializer_ih(weights)
        return lstm

    def forward(self,inputs):
        enc_out, enc_state = self.encoder(inputs)
        dec_out, dec_state = self.decoder(torch.zeros_like(inputs),enc_state)
        return dec_out, dec_state
