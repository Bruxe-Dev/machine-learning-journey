import numpy 

def predict(X,W,b):
    x_poly = np.column_stack([
        x,
        x**2,
        x**3,
        x**4
    ])

    return np.matmul(x_poly,W)+b

X = np.array([1,2,3,4])
Y = np.array([2,5,10,17])
W = np.array([1.0,0.5,0.1])
b = 1.0