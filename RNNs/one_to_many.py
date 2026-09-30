import numpy as np

#from multilabel_tensorflow import output_layer

class OneToMany():
    def __init__(self, n_input,n_hidden,n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.wx_h = np.random.randn(n_hidden,n_input)
        self.wh_h = np.random.randn(n_hidden,n_hidden)

        self.b_h = np.zeros(n_hidden)

        self.wh_y = np.random.randn(n_output,n_hidden)
        self.b_y = np.zeros(n_output)

    def forward (self,x,n_runs):
        h = np.zeros(self.n_hidden)
        outputs = []

        z_h = self.wx_h@x+self.wh_h@h+self.b_h
        h = np.tanh(z_h)
        y = self.wh_y@h+self.b_y

        outputs.append(y)

        return outputs

rnn = OneToMany(
    n_input=3,
    n_hidden=2,
    n_output=4
)

x = np.array([1.0, 2.0, 3.0])

outputs = rnn.forward(x, n_runs=3)

for i, output in enumerate(outputs, start=1):
    print(f"Output {i}:", output)