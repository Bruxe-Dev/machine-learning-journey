import numpy as np

class LSTM():
    def __init__(self,n_input,n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.w_f = np.random.randn(n_hidden, n_hidden + n_input)
        self.b_f = np.zeros(n_hidden)

    def sigmoid(self,z_t):
        return 1 /(1+np.exp(-z_t))

    def forget_gate(self,X,hidden_states_prev):
        x_hidden_prev = np.concat((X,hidden_states_prev),axis=0)

        z_t = self.w_f @ x_hidden_prev + self.b_f

        return self.sigmoid(z_t)

