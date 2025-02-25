import random

inputs = [0.5, 0.10]
biases = [0.5, 0.7]

def get_weight():
    return random.uniform(-0.5, 0.5)

weights = [get_weight() for _ in range(8)]


def exponential(x, terms=20):
    result = 1.0
    term = 1.0
    for n in range(1, terms):
        term *= x / n
        result += term
    return result

def tanh(x):
    if x > 100:  
        return 1.0
    elif x < -100:  
        return -1.0
    e_2x = exponential(2 * x) 
    return (e_2x - 1) / (e_2x + 1)

def forward(inputs, weights, biases):
    h1 = tanh(inputs[0] * weights[0] + inputs[1] * weights[1] + biases[0])
    h2 = tanh(inputs[0] * weights[2] + inputs[1] * weights[3] + biases[0])
    
    o1 = tanh(h1 * weights[4] + h2 * weights[5] + biases[1])
    o2 = tanh(h1 * weights[6] + h2 * weights[7] + biases[1])
    
    return o1, o2


output = forward(inputs, weights, biases)

print("out:", output)
