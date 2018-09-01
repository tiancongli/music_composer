import os

import numpy as np

from data import trainset
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from settings import *


def get_midi_batch(trainset):
    batch_start = 0
    while True:
        batch_end = batch_start + TIME_STEPS * BATCH_SIZE

        # if exceed the trainset, double it
        if batch_end >= trainset.shape[0]:
            # break
            trainset = np.vstack((trainset, trainset))

        origin_seq = trainset[batch_start: batch_end, :].reshape(BATCH_SIZE,
                                                                 TIME_STEPS, -1)
        output_seq = np.zeros((BATCH_SIZE, OUTPUT_SIZE))
        for i in range(BATCH_SIZE):
            output_seq[i] = trainset[batch_start + TIME_STEPS * (i + 1)]

        # use every step as a new start of the train sequence
        batch_start += 1
        yield origin_seq, output_seq


def init_model(batch_size=BATCH_SIZE, weights=None):
    model = Sequential()
    # build a LSTM RNN
    model.add(LSTM(
        batch_input_shape=(batch_size, TIME_STEPS, INPUT_SIZE),
        output_dim=CELL_SIZE,
        # return_sequences=True,      # True: output at all steps. False: output
        # as last step.
        stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
    ))
    # add output layer
    model.add(Dense(OUTPUT_SIZE))

    # set weights manually
    if weights:
        model.set_weights(weights)

    model.compile(optimizer=Adam(LR), loss='mse',)
    return model


################################START TRAINING###########################


if __name__ == "__main__":
    if os.path.isfile(MODEL_NAME):
        model = load_model(MODEL_NAME)
    else:
        model = init_model()
    
    print('------------ Training ------------')
    model.fit_generator(get_midi_batch(trainset), steps_per_epoch=50,
                        epochs=100,
                        verbose=2)
    model.save(MODEL_NAME)
