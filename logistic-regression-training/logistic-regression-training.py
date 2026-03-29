import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for step in range(steps):
        pred = _sigmoid(X@w + b)

        """
        dL/dw_i = d/dw_i [y_i log p_i + (1 - y_i) log(1 - p_i)]

        d/dw_i[log p_i] = 1/N [y_i (1 - p_i)(-X) + (1 - y_i) (p_i)(-X)] = 1/N (-X(p_i - y_i))

        dL/db_i = d/db_i [y_i log p_i + (1 - y_i) log(1 - p_i)]

        d/db_i[log p_i] = 1/N [y_i (1 - p_i) + (1 - y_i) (p_i)] = 1/N(p_i - y_i)
        
        """

        grad_w = X.T@(pred - y) / n
        w = w - grad_w

        grad_b = np.mean(pred - y)
        b = b - grad_b

    return (w, b)