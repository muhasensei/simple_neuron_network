import numpy as np
def act(x):
    return 0 if x < 0.5 else 1

def go(car, mambo, handsome):
    inputs = np.array([car, mambo, handsome])
    w11 = [0.1, 0.3, 0]#non-prioritizer
    w12 = [0.4, -0.5, 1]#prioritizer

    weight1 = np.array([w11, w12])
    weight2 = [-1, 1]

    hidden_layer_values = np.dot(weight1, inputs)
    # print(hidden_layer)

    hidden_layer_outputs = np.array([act(x) for x in hidden_layer_values])
    # print(hidden_layer_outputs)

    res = np.dot(hidden_layer_outputs, weight2)
    return act(res)

decision = go(1, 0, 0)

if decision == 1:
    print('Let`s go')
else:
    print('no, thanks')





