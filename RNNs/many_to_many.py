import numpy as np

#from RNNs.one_to_many import outputs
#from multiclass_numpy import start


class ManyToMany():
    def __init__(self,n_input,n_hidden,n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.wx_h = np.random.randn(n_hidden,n_input)
        self.wh_h = np.random.randn(n_hidden,n_hidden)

        self.b_h = np.zeros(n_hidden)

        self.wh_y  = np.random.randn(n_hidden,n_output)
        self.b_y = np.zeros(n_output)

    def forward_prop(self,X):
        h = np.zeros(self.n_hidden)
        results = []

        for _ in X :
           z_h = self.wx_h@_+self.wh_h@h+self.b_h
           h = np.tanh(z_h)
           y = self.wh_y@h+self.b_y

           results.append(y)

        return results

if __name__ == "__main__":
    mtm_rnn = ManyToMany(
        5,
        7,
        34
    )

    x = np.array([
        [1.0, 2.0, 3.0, 4.0, 6.0],
        [2.0, 3.0, 4.0, 6.0, 5.0],
        [3.0, 4.0, 5.0, 2.0, 3.0]
    ])

    outputs = mtm_rnn.forward_prop(x)

    for i, output in enumerate(outputs,start=1):
        print(f"Output {i} is {output}")