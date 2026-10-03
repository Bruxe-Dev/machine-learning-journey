import numpy as np

class ManyToMany():
    def __init__(self,n_input,n_hidden,n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.wx_h = np.random.randn(n_hidden,n_input)
        self.wh_h = np.random.randn(n_hidden,n_hidden)

        self.b_h = np.zeros(n_hidden)

        self.wh_y  = np.random.randn(n_output,n_hidden)
        self.b_y = np.zeros(n_output)

    def forward_prop(self,X):
        h = np.zeros(self.n_hidden)
        results = []
        hidden_states = []


        for _ in X :
           z_h = self.wx_h@_+self.wh_h@h+self.b_h
           h = np.tanh(z_h)
           y = self.wh_y@h+self.b_y

           results.append(y)
           hidden_states.append(h)

        return hidden_states, results
    def backward_prop(self,outputs,hidden_states,targets):

        for i in range(len(hidden_states)):
            output = outputs[i]
            target = targets[i]
            h = hidden_states[i]

            dy = output - target
            dw_hy = np.outer(dy, h)
            db = dy
            dh = self.wh_y.T@dy

    def loss(self,results,targets):
        results = np.array(results)
        losses = 0.5 * np.sum((targets - results)**2)

        return losses

if __name__ == "__main__":
    mtm_rnn = ManyToMany(
        5,
        7,
        34
    )

    x = np.array([
        [1.0, 2.0, 3.0, 4.0, 6.0],
        [2.0, 3.0, 4.0, 6.0, 5.0],
        [3.0, 1.0, 2.0, 2.0, 3.0]
    ])

    targets = np.array([
        np.ones(34),
        np.ones(34) * 2,
        np.ones(34) * 3
    ])
    hidden_states,outputs = mtm_rnn.forward_prop(x)
    losses = mtm_rnn.loss(outputs,targets)

    print(f"Losses: {losses}")

    for i, output in enumerate(outputs,start=1):
        print(f"Output {i} is {output}")