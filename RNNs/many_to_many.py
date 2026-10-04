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
        inputs = []
        hidden_states = []
        hidden_states_prev = []

        for _ in X :
           hidden_states_prev.append(h.copy())
           z_h = self.wx_h@_+self.wh_h@h+self.b_h
           h = np.tanh(z_h)
           y = self.wh_y@h+self.b_y

           inputs.append(_)
           results.append(y)
           hidden_states.append(h)

        return hidden_states, results,inputs,hidden_states_prev
    def backward_prop(
        self,
        outputs,
        inputs,
        hidden_states,
        targets,
        hidden_states_prev
        ):
        dW_xh = np.zeros_like(self.wx_h)
        dW_hh = np.zeros_like(self.wh_h)
        db_h = np.zeros_like(self.b_h)

        dW_hy = np.zeros_like(self.wh_y)
        db_y = np.zeros_like(self.b_y)

        dh_next  = np.zeros(self.n_hidden)

        for i in reversed(range(len(hidden_states))):
            output = outputs[i]
            target = targets[i]
            h = hidden_states[i]

            dy = output - target

            dW_hy += np.outer(dy, h)
            db_y += dy

            dh = self.wh_y.T @ dy + dh_next

            dz = dh * (1 - h**2)

            dW_xh += np.outer(dz, inputs[i])
            dW_hh += np.outer(dz, hidden_states_prev[i])
            db_h += dz

            dh_next = self.wh_h.T @ dz

        return dW_xh,dW_hh,dW_hy,db_h,db_y

    def update_weight(
        self,
        dW_xh,
        dW_hh,
        dW_hy,
        db_h,
        db_y,
        learning_rate,
    ):
        self.wx_h -= learning_rate * dW_xh
        self.wh_h -= learning_rate * dW_hh
        self.wh_y -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

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
    epochs = 2000
    learning_rate = 0.001
    for epoch in range(epochs):

        hidden_states, outputs, inputs, hidden_states_prev = mtm_rnn.forward_prop(x)

        loss = mtm_rnn.loss(outputs, targets)

        gradients = mtm_rnn.backward_prop(
            outputs,
            inputs,
            hidden_states,
            targets,
            hidden_states_prev
        )

        dW_xh, dW_hh, dW_hy, db_h, db_y = gradients

        mtm_rnn.update_weight(
            dW_xh,
            dW_hh,
            dW_hy,
            db_h,
            db_y,
            learning_rate
        )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")