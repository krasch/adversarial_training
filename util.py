import json

from keras.callbacks import Callback


class JSONLogger(Callback):

    samples_seen = 0

    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        logs = {key: value for key, value in logs.items() if key not in ["size", "batch"]}
        logs["samples"] = self.samples_seen
        logs["epoch"] = epoch
        print(json.dumps(logs))

    def on_batch_end(self, batch, logs=None):
        self.samples_seen += logs["size"]

        if batch % 10 == 0:
            logs = {key:value for key, value in logs.items() if key not in ["size", "batch"]}
            logs["samples"] = self.samples_seen
            print(json.dumps(logs))