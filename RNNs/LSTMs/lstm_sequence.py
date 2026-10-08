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

        c_t = f_t * self.c_prev + i_t * candidate

        h_t = o_t * np.tanh(c_t)

        self.c_prev = c_t
        self.h_prev = h_t

        return c_t,h_t

    def forward(self,X):
        cell_states = []
        hidden_states = []

        for x_t in X:
            c_t,h_t = self.forward_step(x_t)

            cell_states.append(c_t)
            hidden_states.append(h_t)

        return np.array(cell_states),np.array(hidden_states)

if __name__ == "__main__":
    lstm = LSTM(
        n_input=3,
        n_hidden=4
    )

    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],

    ])

    cell_states,hidden_states = lstm.forward(X)

    print(f"The Cell States are: {cell_states}")
    print(f"The Hidden states: {
        hidden_states
    }")