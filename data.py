import os

import numpy as np

import midi
from sklearn.preprocessing import MinMaxScaler


def get_trainset():
    res = []
    for file in os.listdir('.'):
        if file.endswith('.mid'):
            print 'load file: ' + file
            pattern = midi.read_midifile(file)
            # only use tick, pitch and velocity as features at this moment
            tick_list = []
            pitch_list = []
            velocity_list = []

            for event in pattern[1]:
                if isinstance(event, midi.NoteOnEvent):
                    # only use NoteOnEvent
                    tick_list.append(event.tick)
                    pitch_list.append(event.data[0])
                    velocity_list.append(event.data[1])

            for i in range(len(tick_list)):
                time_step = [tick_list[i], pitch_list[i], velocity_list[i]]
                res.append(time_step)

    np_res = np.array(res)
    print "get trainset with shape: " + str(np_res.shape)
    return np_res


# get trainset
scaler = MinMaxScaler()
trainset = scaler.fit_transform(get_trainset())
