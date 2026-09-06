import numpy as np

X = np.array([1,2,3,4,5])
Y = np.array([2,5,10,17,24])
W = np.array([1.0,0.5,0.1,0.3])
b = 1.0

lambda_ = 1.0

def compute_cost_l2(x_poly,y,W,b):
    m = x_poly.shape[0]

    predictions = np.matmul(x_poly,W)+b

    error = predictions - y

    l2_penalty = (lambda_ / (2 * m)) * np.sum(W ** 2)
    ordinary_cost = (1/(2 * m)) * np.sum(error ** 2)

    cost = ordinary_cost + l2_penalty

    return cost

x_poly = np.column_stack([
    X,
    X**2,
    X**3,
    X**4
])


cost = compute_cost_l2(x_poly,Y,W,b)

print(f"Cost: {cost}")