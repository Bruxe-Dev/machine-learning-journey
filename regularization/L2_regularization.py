import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 5, 10, 17, 24])

W = np.array([1.0, 0.5, 0.1, 0.3])
b = 1.0

lambda_ = 1.0


def compute_cost_l2(x_poly, y, W, b, lambda_):
    m = x_poly.shape[0]

    predictions = np.matmul(x_poly, W) + b

    error = predictions - y

    ordinary_cost = (1 / (2 * m)) * np.sum(error ** 2)

    l2_penalty = (lambda_ / (2 * m)) * np.sum(W ** 2)

    cost = ordinary_cost + l2_penalty

    return cost


def compute_gradient_l2(x_poly, y, W, b, lambda_):
    m = x_poly.shape[0]

    predictions = np.matmul(x_poly, W) + b

    errors = predictions - y

    dj_dw = (1 / m) * np.matmul(x_poly.T, errors)

    dj_dw += (lambda_ / m) * W

    dj_db = (1 / m) * np.sum(errors)

    return dj_dw, dj_db


x_poly = np.column_stack([
    X,
    X**2,
    X**3,
    X**4
])


cost = compute_cost_l2(
    x_poly,
    Y,
    W,
    b,
    lambda_
)

dj_dw, dj_db = compute_gradient_l2(
    x_poly,
    Y,
    W,
    b,
    lambda_
)

print(f"Cost: {cost}")
print(f"dj_dw: {dj_dw}")
print(f"dj_db: {dj_db}")