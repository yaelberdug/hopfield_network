# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import argparse

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def getOptions():
    parser = argparse.ArgumentParser(description='Parses Command.')
    parser.add_argument('-t', '--train', nargs='*', help='Training data directories.')
    parser.add_argument('-i', '--iteration', type=int, help='Number of iteration.')
    parser.add_argument('-p', '--predict', nargs='*', help='Predict image.')
    parser.add_argument('-s', '--size', type=int, help='Image size nXn.')
    options = parser.parse_args(sys.argv[1:])
    return options



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.random.seed(1)
    options = getOptions()
    input_shape = (options.size, options.size)
    model = hopfield(input_shape)
    print('Model initialized with weights shape ', model.W.shape)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
