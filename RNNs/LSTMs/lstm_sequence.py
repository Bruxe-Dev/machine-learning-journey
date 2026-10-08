import numpy as np 

def sigmoid(z_t):
    return 1/(1 + np.exp(-z_t))

class LSTM():
    def __init__(self,n_input,n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.h_prev = np.zeros(n_hidden)
        self.c_prev = np.zeros(n_hidden)

        self.w_i = np.random.randn(n_hidden, n_input+n_hidden)
        self.w_f = np.random.randn(n_hidden, n_input+n_hidden)
        self.w_c = np.random.randn(n_hidden,n_input+n_hidden)
        self.w_o = np.random.randn(n_hidden,n_input+n_hidden)

        self.b_i = np.zeros(n_hidden)
        self.b_f = np.zeros(n_hidden)
        self.b_c = np.zeros(n_hidden)
        self.b_o = np.zeros(n_hidden)

    def forward_step(self,X):
        x_hidden_prev = np.concatenate((X,self.h_prev),axis=0)

        i_t = sigmoid(self.w_i @ x_hidden_prev + self.b_i)
        f_t = sigmoid(self.w_f @ x_hidden_prev + self.b_f)
        candidate = np.tanh(self.w_c @ x_hidden_prev + self.b_c)
        o_t = sigmoid(self.w_o @ x_hidden_prev + self.b_o)