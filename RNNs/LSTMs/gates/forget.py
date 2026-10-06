import numpy as np

class LSTM():
    def __init__(self,n_input,n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.w_c = np.random.randn(n_hidden, n_hidden + n_input)
        self.w_i = np.random.randn(n_hidden, n_hidden + n_input)
        self.w_f = np.random.randn(n_hidden, n_hidden + n_input)
        self.w_o = np.random.randn(n_hidden, n_hidden + n_input)
        self.b_c = np.zeros(n_hidden)
        self.b_i = np.zeros(n_hidden)
        self.b_f = np.zeros(n_hidden)
        self.b_o = np.zeros(n_hidden)

    def sigmoid(self,z_t):
        return 1 /(1+np.exp(-z_t))

    def forget_gate(self,X,hidden_states_prev):
        x_hidden_prev = np.concat((X,hidden_states_prev),axis=0)

        z_t = self.w_f @ x_hidden_prev + self.b_f

        return self.sigmoid(z_t)

    def input_gate(self,X,hidden_states_prev):
        x_hidden_prev = np.concat((X,hidden_states_prev),axis=0)

        z_t = self.w_i @ x_hidden_prev +self.b_i
        return self.sigmoid(z_t)

    def content_cell(self,X,hidden_states_prev):
        x_hidden_prev = np.concat((X,hidden_states_prev),axis=0)

        z_t = self.w_c @ x_hidden_prev + self.b_c
        return np.tanh(z_t)

    def cell_state(self,forget_values,input_values,candidate_values,cell_state_prev):

        old_memory = forget_values*cell_state_prev
        new_memory = input_values*candidate_values

        return old_memory + new_memory

    def output_gate(self,X,hidden_states_prev):
        x_hidden_prev = np.concat((X,hidden_states_prev),axis=0)

        z_t = self.w_o @ x_hidden_prev + self.b_o
        return self.sigmoid(z_t)

if __name__ == "__main__":
    lstm = LSTM(n_input=3,n_hidden=4)

    X = np.array([1.0, 2.0, 3.0])
    hidden_prev = np.array([0.5, 0.2, 0.1, 0.4])
    cell_prev = np.zeros(4)
    
    cell_state_values = np.round(lstm.content_cell(X,hidden_prev),4)
    input_values = np.round(lstm.input_gate(X,hidden_prev),4)
    forget_values = np.round(lstm.forget_gate(X,hidden_prev),4)

    cell_values = lstm.cell_state(
        forget_values,
        input_values,
        cell_state_values,
        cell_prev
    )
    output_values = np.round(lstm.output_gate(X,hidden_prev),4)
    
    print(f"Input gate: {input_values}")
    print(f"Cell Value: {input_values}")
    print(f"Cell state: {cell_state_values}")
    print(f"Forget gate: {forget_values}")

    h_t = output_values * np.tanh(cell_state_values)

    print(f"Output values: {h_t}")