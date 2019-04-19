import numpy as np


class data(object):
    def __init__(self,params):
        num_samples = params.num_samples
        batch_size = params.batch_size
        seq_length = params.seq_length 
        dim = params.dim
        train_val_test_split = params.train_val_test_split
        random_seed = params.seed


        assert dim==1,"we doing coinflips!"
        assert num_samples>batch_size,"to batch or not to batch"
        assert sum(train_val_test_split)==1.0, 'what kinda splits you smoking?'
        
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dim = dim
        
        self.train_val_test_split = train_val_test_split
        self.train_size = int(train_val_test_split[0]*num_samples)
        self.val_size = int(train_val_test_split[1]*num_samples)
        self.test_size = int(train_val_test_split[2]*num_samples)

        self.seed = random_seed
        self.get_random_flips(num_samples,batch_size,seq_length,dim,train_val_test_split,random_seed)
        self.get_pseudo_flips(num_samples,batch_size,seq_length,dim,train_val_test_split) 
    
    def get_random_flips(self,num_samples,batch_size,seq_length,dim,train_val_test_split,seed):

        data = np.random.RandomState(seed=seed).randint(0,1+dim,num_samples)
        train = data[:self.train_size].reshape(-1,batch_size,seq_length,dim)
        val = data[self.train_size:self.val_size].reshape(-1,batch_size,seq_length,dim)
        test = data[-self.test_size:].reshape(-1,batch_size,seq_length,dim)

        xtrain = train
        ytrain = xtrain.copy()
        np.random.RandomState(seed=seed).shuffle(ytrain)

        xval = val
        yval = val.copy()
        np.random.RandomState(seed=seed).shuffle(yval)
        
        xtest = test
        ytest = xtest.copy()
        np.random.RandomState(seed=seed).shuffle(ytest)

        self.r_xtrain = xtrain
        self.r_ytrain = ytrain
        self.r_xval = xval
        self.r_yval = yval
        self.r_xtest = xtest
        self.r_ytest = ytest

    def get_pseudo_flips(self,num_samples,batch_size,seq_length,dim,train_val_test_split):
        #psuedo flips have equal distribution but there is an obvious pattern 
        # for example [0,1,0,1,....]

        train = np.array([ 1 if i%2==0 else 0 for i in range(self.train_size)]).reshape(-1,batch_size,seq_length,dim)
        val = np.array([ 1 if i%2==0 else 0 for i in range(self.val_size)]).reshape(-1,batch_size,seq_length,dim)
        test = np.array([ 1 if i%2==0 else 0 for i in range(self.test_size)]).reshape(-1,batch_size,seq_length,dim)

        self.p_xtrain = train
        self.p_ytrain = (train+1)%2
        self.p_xval = val
        self.p_yval = (val+1)%2
        self.p_xtest = test
        self.p_ytest = (test+1)%2


if __name__ == "__main__":
    #test 
    class args(object):
        def __init__(self):
            self.batch_size = 250
            self.num_samples = 100000
            self.seq_length = 10
            self.dim = 1
            self.train_val_test_split = [.6,.2,.2] #going to mix in some non-random sequences duh!
            self.seed = 0 
            
    params = args()

    d  = data(params)
