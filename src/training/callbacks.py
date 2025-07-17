import csv
import gc
import os

from tensorflow.keras import callbacks, backend


class CsvLoggerCallback(callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            if epoch == 0:
                # Write header on first epoch
                writer.writerow(["epoch"] + list(logs.keys()))
            writer.writerow([epoch] + list(logs.values()))

class ValidationAccuracyThresholdCallback(callbacks.Callback):
    def __init__(self, threshold):
        super(ValidationAccuracyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_accuracy") >= self.threshold:
            self.model.stop_training = True


class ClearMemory(callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        backend.clear_session()
        gc.collect()



def get_callbacks():
    # defines when the model will stop training
    accuracy_threshold_callback = ValidationAccuracyThresholdCallback(
        threshold=0.98)

    # # saves a snapshot of the model while it is training
    # checkpoint_filepath = os.path.join(fp["MODEL"], "checkpoint.keras")
    # checkpoint_callback = callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath, save_weights_only=False, save_best_only=True,
    #     monitor='val_loss',
    #     mode='min'
    # )

    # # logs the epoch, accuracy, and loss for a training session
    # csv_logger_callback = CsvLoggerCallback(
    #     os.path.join(fp["MODEL"], "training_logs.csv")
    # )

    # Define the ReduceLROnPlateau callback
    reduce_lr_callback = callbacks.ReduceLROnPlateau(
        monitor="val_loss",  # Metric to monitor
        factor=0.2,  # Factor by which the learning rate will be reduced
        patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=0.00001,  # Lower bound on the learning rate
    )

    clear_memory_callback = ClearMemory()

    return [accuracy_threshold_callback, reduce_lr_callback]
