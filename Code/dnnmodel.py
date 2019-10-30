import tflearn
import constants
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression




def bulidModel(input_size):
    #input_size - size of our state(4)
    network = input_data(shape=[None,input_size, 1], name='input')

    network = fully_connected(network, constants.HL_1, activation='relu')
    # to solve the over fitting problem
    network = dropout(network, constants.keepRatio)

    network = fully_connected(network, constants.HL_2, activation='relu')
    network = dropout(network, constants.keepRatio)

    network = fully_connected(network, constants.HL_3, activation='relu')
    network = dropout(network, constants.keepRatio)

    network = fully_connected(network, constants.HL_4, activation='relu')
    network = dropout(network, constants.keepRatio)

    network = fully_connected(network, constants.HL_5, activation='relu')
    network = dropout(network, constants.keepRatio)

    network = fully_connected(network, constants.OUTPUT_L, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=constants.LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

