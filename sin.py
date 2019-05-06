import mlp
import numpy as np
import json

#Alessandro Baccin - 16724489. alessandro.baccin@ucdconnect.ie
#sin(x1-x2+x3-x4) training and test

def train(inputs, outputs, NN, max_epochs):
    learn_r = 0.025
    for e in range(max_epochs):
        mse = 0

        for p in range(len(inputs)):
            NN.feedforward(inputs[p])
            i = NN.backpropagation(outputs[p])
            mse += (np.square(i)).mean(axis=None)
        NN.update_weights(learn_r)
        mse /= len(inputs)

        if((e + 1) % 250 == 0):
                learn_r -= 0.005

        if((e + 1) % 100 == 0):
            print("Error at epoch e: {} is: {}".format((e + 1), mse))

def read_input_file():
    with open('input_set.json') as file:
        return json.load(file)

train_and_test = read_input_file()                              #Data is retrieved from input_set.json

train_set_inputs = list(train_and_test.values())[0:150]
train_set_outputs = list(map(float, train_and_test.keys()))[0:150]
test_set_inputs = list(train_and_test.values())[150:200]
test_set_outputs = list(map(float, train_and_test.keys()))[150:200]

NN_t_sin_four = mlp.MLP(4, 15, 1, 'tanh')

train(train_set_inputs, train_set_outputs, NN_t_sin_four, 1000)
NN_t_sin_four.test(test_set_inputs, test_set_outputs)