import numpy as np

from logistic_numpy import forward_prop


class ManyToOne():
    def __init__(self,n_input,n_hidden,n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.wx_h = np.random.randn(n_hidden,n_hidden)
        self.wh_h = np.random.randn(n_hidden,n_hidden)

        self.b_h = np.zeros(n_hidden)

        self.wh_y = np.random.randn(n_output,n_hidden)
        self.b_y = np.random.randn(n_output)


    def forward(self,X):
         h = np.zeros(self.n_hidden)

         for x in X :
             z_h = self.wx_h@x+self.wh_h@h+self.b_h
             h = np.tanh(z_h)
         y = self.wh_y@h+self.b_y
         return y

if __name__ == "__main__":
    rnn_model = ManyToOne(5,7,1)
    x = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0]
    ])

    y = rnn_model.forward(x)
    print(f"Predicted Value {y}")