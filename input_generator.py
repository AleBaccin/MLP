import json
import numpy as np

#Alessandro Baccin - 16724489. alessandro.baccin@ucdconnect.ie
#This small script is just to generate the 150+50 elements for training and testing.

def generate_random_set(size):
    test_set = {}
    for _ in range(size):
        current_input = []
        for _ in range(4):
            x = np.random.uniform(-1, 1)
            current_input.append(x)
        test_set[np.sin(current_input[0] - current_input[1] + current_input[2] - current_input[3])] = current_input
    return test_set

input_set = generate_random_set(200)

def write_to_json(dict_):
    with open('input_set.json', 'w') as file:
        file.write(json.dumps(dict_))

write_to_json(input_set)