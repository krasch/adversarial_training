import json

from keras.callbacks import Callback
import numpy as np


def dump_json(some_dict):
    for key in some_dict:
        if isinstance(some_dict[key], np.float32):
            some_dict[key] = float(some_dict[key])
    return json.dumps(some_dict)


class JSONLogger(Callback):

    samples_seen = 0

    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        logs = {key: value for key, value in logs.items() if key not in ["size", "batch"]}
        logs["samples"] = self.samples_seen
        logs["epoch"] = epoch
        print(dump_json(logs))

    def on_batch_end(self, batch, logs=None):
        self.samples_seen += logs["size"]

        if batch % 10 == 0:
            logs = {key: value for key, value in logs.items() if key not in ["size", "batch"]}
            logs["samples"] = self.samples_seen
            print(dump_json(logs))