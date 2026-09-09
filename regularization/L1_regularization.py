import numpy as np 

def compute_lasso_cost(X, y, w, b, lambda_):
    m = len(y)

    y_pred = w @ x + b

    errors = y - y_pred

    mse = (1/(2 * m)) * np.sum(errors ** 2)

    l1_penalty = (lambda_ / m) *np.sum(np.abs(w))

    cost = mse + l1_penalty

    return cost 

    