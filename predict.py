import numpy as np

from data import scaler, trainset
from keras.engine.saving import load_model
from settings import *
from train import init_model


def get_head():
    # get initial sequence for prediction
    return trainset[: TIME_STEPS, :].reshape(1, TIME_STEPS, -1)


def get_model_for_prediction():
    # reset batchsize to 1 of the trained model
    trained_model = load_model(MODEL_NAME)
    return init_model(batch_size=1, weights=trained_model.get_weights())


def generate_notes(model, size):
    res = np.zeros((size, OUTPUT_SIZE))
    input = get_head()
    for i in range(size):
        output = model.predict(input, batch_size=1)
        input = np.concatenate((input[:, 1:TIME_STEPS, :], output.reshape(
            1, 1, OUTPUT_SIZE)), axis=1)
        res[i] = output
    return res


if __name__ == "__main__":
    model = get_model_for_prediction()

    res = generate_notes(model, 100)
    final_res = scaler.inverse_transform(res).astype(int)
    print final_res
