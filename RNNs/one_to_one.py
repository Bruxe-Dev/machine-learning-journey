import numpy as np

class OneToOne():
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.wx_h = np.random.randn(hidden_size,input_size)
        self.b_h = np.zeros(hidden_size)
        
        self.w_hy = np.random.randn(output_size,hidden_size)
        self.b_y = np.zeros(output_size)
    
    def foward(self,x, wx_h=None, b_h=None, w_hy=None, b_y=None):
        z_h = self.wx_h@x+self.b_h
        h = np.tanh(z_h)
        y = self.w_hy@h+self.b_y
    
        return h,y

if __name__ == "__main__":
    rnn = OneToOne(3,2,4)

    x = np.array([1.0,2.0,3.0])
    h,y = rnn.foward(x)
    print("Hidden:", h)
    print("Output:", y)