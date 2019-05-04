import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

class CompressData(Dataset):
    def __init__(self,p_bias, seq_len, num_epochs):
        self.p_bias = p_bias
        self.seq_len = seq_len
        self.num_epochs = num_epochs
    def __len__(self):
        return self.num_epochs
    def __getitem__(self,idx):
        x = torch.rand(self.seq_len,1)
        x = x < self.p_bias
        return x




class Data(object):
    def __init__(self,params,seq_length):
        num_samples = params.num_samples
        batch_size = params.batch_size
        dim = params.dim
        train_val_test_split = params.train_val_test_split
        random_seed = params.seed
        p_bias = params.p_bias


        assert dim==1,"we doing coinflips!"
        assert num_samples>batch_size,"to batch or not to batch"
        assert sum(train_val_test_split)==1, 'what kinda splits you smoking?'
        
        self.batch_size = batch_size
        self.num_samples = num_samples*seq_length
        self.dim = dim
        self.p_bias = p_bias
        self.seq_length = seq_length
        
        self.train_val_test_split = train_val_test_split
        self.train_size = int(train_val_test_split[0]*self.num_samples)
        self.val_size = int(train_val_test_split[1]*self.num_samples)
        self.test_size = int(train_val_test_split[2]*self.num_samples)

        self.seed = random_seed
        self.get_random_flips(self.num_samples,batch_size,seq_length,dim,train_val_test_split,random_seed)
        self.get_pseudo_flips(self.num_samples,batch_size,seq_length,dim,train_val_test_split) 
    
    def get_random_flips(self,num_samples,batch_size,seq_length,dim,train_val_test_split,seed):

        if self.p_bias == 0.5:
            data = np.random.RandomState(seed=seed).randint(0,1+dim,num_samples)
        else:
            data = np.random.RandomState(seed=seed).random(num_samples)
            data = data < self.p_bias
            data = data.astype(np.int64)
        train = data[:self.train_size].reshape(-1,batch_size,seq_length,dim).swapaxes(2,1)
        val = data[self.train_size:self.val_size].reshape(-1,batch_size,seq_length,dim).swapaxes(2,1)
        test = data[-self.test_size:].reshape(-1,batch_size,seq_length,dim).swapaxes(2,1)

        xtrain = train
        ytrain = xtrain.copy()
        np.random.RandomState(seed=seed).shuffle(ytrain)

        xval = val
        yval = val.copy()
        np.random.RandomState(seed=seed).shuffle(yval)
        
        xtest = test
        ytest = xtest.copy()
        np.random.RandomState(seed=seed).shuffle(ytest)

        self.random_xtrain = torch.from_numpy(xtrain).float()
        self.random_ytrain = torch.from_numpy(ytrain).float()
        self.random_xval =   torch.from_numpy(xval).float()
        self.random_yval =   torch.from_numpy(yval).float()
        self.random_xtest =  torch.from_numpy(xtest).float()
        self.random_ytest =  torch.from_numpy(ytest).float()

    def get_pseudo_flips(self,num_samples,batch_size,seq_length,dim,train_val_test_split):
        #psuedo flips have equal distribution but there is an obvious pattern 
        # for example [0,1,0,1,....]

        train = np.array([ 1 if i%2==0 else 0 for i in range(self.train_size)]).reshape(-1,batch_size,seq_length,dim).swapaxes(2,1)
        val = np.array([ 1 if i%2==0 else 0 for i in range(self.val_size)]).reshape(-1,batch_size,seq_length,dim).swapaxes(2,1)
        test = np.array([ 1 if i%2==0 else 0 for i in range(self.test_size)]).reshape(-1,batch_size,seq_length,dim).swapaxes(2,1)

        self.pseudo_xtrain =  torch.from_numpy(train).float()
        self.pseudo_ytrain =  torch.from_numpy((train+1)%2).float()
        self.pseudo_xval =    torch.from_numpy(val).float()
        self.pseudo_yval =    torch.from_numpy((val+1)%2).float()
        self.pseudo_xtest =   torch.from_numpy(test).float()
        self.pseudo_ytest =   torch.from_numpy((test+1)%2).float()

# if __name__ == "__main__":
#     #test 
#     class args(object):
#         def __init__(self):
#             self.batch_size = 250
#             self.num_samples = 100000
#             self.seq_length = 10
#             self.dim = 1
#             self.train_val_test_split = [.6,.2,.2] #going to mix in some non-random sequences duh!
#             self.seed = 0 
            
#     params = args()

#     d  = Data(params)

# if __name__ == "__main__":
#     data = CompressData(.5,10,2)
#     a = np.random.randn(2,5)
#     b = collate_fn(a)
