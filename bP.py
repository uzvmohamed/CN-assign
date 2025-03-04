import random
import math

def get_weight():
    return random.uniform(-0.5, 0.5)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward(inputs, weights, biases):
    h1_net = inputs[0] * weights[0] + inputs[1] * weights[1] + biases[0]
    h2_net = inputs[0] * weights[2] + inputs[1] * weights[3] + biases[0]
    h1 = sigmoid(h1_net)
    h2 = sigmoid(h2_net)
    
    o1_net = h1 * weights[4] + h2 * weights[5] + biases[1]
    o2_net = h1 * weights[6] + h2 * weights[7] + biases[1]
    o1 = sigmoid(o1_net)
    o2 = sigmoid(o2_net)
    
    return h1, h2, o1, o2, h1_net, h2_net, o1_net, o2_net

def backward(inputs, weights, biases, target, learning_rate=0.5):
    h1, h2, o1, o2, h1_net, h2_net, o1_net, o2_net = forward(inputs, weights, biases)
    
    error_o1 = 0.5 * (target[0] - o1) ** 2
    error_o2 = 0.5 * (target[1] - o2) ** 2
    total_error = error_o1 + error_o2
    
    delta_o1 = (o1 - target[0]) * sigmoid_derivative(o1)
    delta_o2 = (o2 - target[1]) * sigmoid_derivative(o2)
    
    delta_h1 = (delta_o1 * weights[4] + delta_o2 * weights[6]) * sigmoid_derivative(h1)
    delta_h2 = (delta_o1 * weights[5] + delta_o2 * weights[7]) * sigmoid_derivative(h2)
    
    weights[0] -= learning_rate * delta_h1 * inputs[0]
    weights[1] -= learning_rate * delta_h1 * inputs[1]
    weights[2] -= learning_rate * delta_h2 * inputs[0]
    weights[3] -= learning_rate * delta_h2 * inputs[1]
    
    weights[4] -= learning_rate * delta_o1 * h1
    weights[5] -= learning_rate * delta_o1 * h2
    weights[6] -= learning_rate * delta_o2 * h1
    weights[7] -= learning_rate * delta_o2 * h2
    
    # Update biases
    biases[0] -= learning_rate * (delta_h1 + delta_h2)
    biases[1] -= learning_rate * (delta_o1 + delta_o2)
    
    return weights, biases, total_error

inputs = [0.05, 0.10]
biases = [0.35, 0.60]
weights = [get_weight() for _ in range(8)]
targets = [0.01, 0.99]  

for _ in range(10000):
    weights, biases, error = backward(inputs, weights, biases, targets)
    if _ % 1000 == 0:
        print(f"Iteration {_}, Error: {error}")

output = forward(inputs, weights, biases)
print("Final Output:", output[2], output[3])
