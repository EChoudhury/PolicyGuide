import torch
import pickle
from pytorch_lightning import Callback

class NanDetectorCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if the training loss is NaN
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs.get("loss", None)
        if loss is not None and torch.isnan(loss):
            batch_file = f"nan_batch_{batch_idx}.pkl"
            with open(batch_file, "wb") as f:
                pickle.dump(batch, f)
            print(f"NaN detected at batch {batch_idx}. Batch saved as {batch_file}")

        # Check for NaNs or Infs in each tensor of the batch
        anomalies = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    anomalies[key] = "contains NaNs"
                if torch.isinf(value).any():
                    anomalies[key] = "contains Infs"
                if not value.all():
                    anomalies[key] = "contains at least one zero"
        if anomalies:
            print(f"Anomalies in batch {batch_idx}: {anomalies}")