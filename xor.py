import mlp
import numpy as np

#Alessandro Baccin - 16724489. alessandro.baccin@ucdconnect.ie
#Xor training and test

def train(inputs, outputs, NN, max_epochs):
    for e in range(max_epochs):
        mse = 0
        learn_r = 0.1

        for p in range(len(inputs)):
            NN.feedforward(inputs[p])
            i = NN.backpropagation(outputs[p])
            mse += (np.square(i)).mean(axis=0)
        NN.update_weights(learn_r)
        mse /= len(inputs)

        if((e + 1) % 200 == 0):
            print("Error at epoch e: {} is: {}".format((e + 1), mse))

inputs=np.array([[0,0], [0,1], [1,0], [1,1]])

outputs=np.array([ -1, 1, 1, -1 ])                                     #tanh spans a different range

NN_t_xor = mlp.MLP(2, 5, 1, 'tanh')

train(inputs, outputs, NN_t_xor, 1000)

NN_t_xor.test(inputs, outputs)