import pandas as pd
import numpy as np
c_params = [1,2]

n_users= 10
num_features = 3
x = np.random.randn(n_users, num_features)

context_df = pd.DataFrame(x, columns=range(num_features))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reward_func_action(params, x, a):
    prob = sigmoid(x[0] + x[1] * params[0] + a * x[2] * params[1] * a**2)
    return np.random.binomial(1, prob)


a = np.linspace(0, 1, 100)
context_df.apply(lambda x: reward_func(c_params, x, 1), axis=1)

