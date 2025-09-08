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

class EarlyStoppingByValThreshold(callbacks.Callback):
    def __init__(self, monitor='val_accuracy', threshold=0.95, mode='greater'):
        super(EarlyStoppingByValThreshold, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        if mode not in ['greater', 'less']:
            raise ValueError("mode must be 'greater' or 'less'")
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: Metric '{self.monitor}' is not available. Available metrics: {list(logs.keys())}")
            return
        
        if (self.mode == 'greater' and current >= self.threshold) or \
           (self.mode == 'less' and current <= self.threshold):
            print(f"\nEpoch {epoch + 1}: Reached {self.monitor} threshold of {self.threshold}. Stopping training.")
            self.model.stop_training = True

class ClearMemory(callbacks.Callback): # TODO depreciated
    def on_epoch_end(self, batch, logs=None):
        backend.clear_session()
        gc.collect()

def get_callbacks(keras_model_dir: str, checkpoint_name: str, stopping_threshold: float) -> list[callbacks.Callback]:
    # defines when the model will stop training
    accuracy_threshold_callback = EarlyStoppingByValThreshold(
            monitor='val_categorical_accuracy',
            threshold=stopping_threshold,
            )

    # saves a snapshot of the model while it is training
    checkpoint_path = os.path.join(keras_model_dir, checkpoint_name)
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=False, save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    # logs the epoch, accuracy, and loss for a training session
    csv_path = os.path.join(keras_model_dir, "training_logs.csv")
    csv_logger_callback = CsvLoggerCallback(csv_path)

    return [accuracy_threshold_callback, checkpoint_callback, csv_logger_callback]
