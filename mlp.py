import numpy as np

#Alessandro Baccin - 16724489. alessandro.baccin@ucdconnect.ie
#Multilayer Perceptron, the structure similar to the one suggested in the assignment description

def sigmoid (x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1.0 - x)

def tanh_prime(x):
    return 1 - np.power(np.tanh(x), 2)

class MLP:
    def __init__(self, n_inputs, n_hidden_units, n_outputs, type_of_act_f):
        self.input = 0
        self.type_of_act_f = type_of_act_f
        self.act_f = 0
        self.act_f_prime = 0

        self.NI = n_inputs
        self.NH = n_hidden_units
        self.NO = n_outputs

        self.Wl = np.zeros((self.NI, self.NH))
        self.Wu = np.zeros((self.NH, self.NO))

        self.dWl = np.zeros_like(self.Wl)
        self.dWu = np.zeros_like(self.Wu)

        self.Zl = np.zeros((1, self.NH))
        self.Zu = np.zeros((1, self.NO))            #Biases

        self.dZl = np.zeros_like(self.Zl)
        self.dZu = np.zeros_like(self.Zu)

        self.H = np.zeros(self.NH)
        self.O = np.zeros(self.NO)

        self.randomise()
        self.set_function()

    def set_function(self):
        if(self.type_of_act_f == 'sig'):
            self.act_f = sigmoid
            self.act_f_prime = sigmoid_prime
        elif(self.type_of_act_f == 'tanh'):
            self.act_f = np.tanh
            self.act_f_prime = tanh_prime

    def feedforward(self, input):
        self.input = np.array(input).reshape(1, self.NI)
        self.H = self.input.dot(self.Wl) + self.Zl
        self.H = self.act_f(self.H)
        self.O = self.H.dot(self.Wu) + self.Zu
        self.O = self.act_f(self.O)

    def randomise(self):
        self.Wl = np.random.uniform(0.1, 0.3, (self.NI, self.NH))
        self.Wu = np.random.uniform(0.1, 0.3, (self.NH, self.NO))
        self.dWl = np.zeros_like(self.Wl)
        self.dWu = np.zeros_like(self.Wu)
        
    def update_weights(self, learning_rate):
        self.Wl += learning_rate*self.dWl
        self.Wu += learning_rate*self.dWu
        self.Zl += learning_rate*self.dZl
        self.Zu += learning_rate*self.dZu

        self.dWl = np.zeros_like(self.Wl)
        self.dWu = np.zeros_like(self.Wu)
        self.dZl = np.zeros_like(self.Zl)
        self.dZu = np.zeros_like(self.Zu)

    def backpropagation(self, target):
        target = np.array(target)
        outer_error = target - self.O
        gradient = outer_error * self.act_f_prime(self.O)

        hidden_error = gradient.dot(self.Wu.T)
        hidden_gradient = hidden_error * self.act_f_prime(self.H)

        self.dWl += self.input.T.dot(hidden_gradient)
        self.dWu += self.H.T.dot(gradient)

        self.Zl += hidden_gradient
        self.Zu += gradient

        return outer_error

    def test(self, inputs, outputs):
        mse = 0
        for c in range(len(inputs)):
            self.feedforward(inputs[c])
            e = (np.square(outputs[c] - np.array(self.O))).mean(axis=0)
            mse += e

            if(len(inputs) > 10):
                if(c % 5 == 0):
                    print("+Input: {}\n-Target: {} Obtained: {}\n-->Squared Error compared with target: {}".format(inputs[c], outputs[c], self.O, e))
            else:
                print("+Input: {}\n-Target: {} Obtained: {}\n-->Squared Error compared with target: {}".format(inputs[c], outputs[c], self.O, e))
        
        mse /= len(inputs)
        print("----->Average Squared Error: {}".format(mse))






