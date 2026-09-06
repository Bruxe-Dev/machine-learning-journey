import numpy 

def compute_cost(x_poly,y,W,b):
    m = x_poly.shape[0]

    predictions = np.matmul(x_poly,W)+b

    error = predictions - y

    cost = (1/(2 * m)) * np.sum(error **2)

    return cost

X = np.array([1,2,3,4,5])
Y = np.array([2,5,10,17,24])
W = np.array([1.0,0.5,0.1,0.3])
b = 1.0

x_poly = np.column_stack([
    x,
    x**2,
    x**3,
    x**4
])


cost = compute_cost(x_poly,Y,W,b)

print(f"Cost: {cost}")