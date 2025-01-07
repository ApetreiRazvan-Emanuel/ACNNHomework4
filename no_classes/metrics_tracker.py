import time
import torch
import pandas as pd
from torch.cuda import max_memory_allocated, reset_peak_memory_stats


class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        self.start_time = None
        reset_peak_memory_stats()

    def start_training(self):
        self.start_time = time.time()

    def update_metrics(self, epoch, train_loss, train_acc, test_loss, test_acc):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_acc'].append(test_acc)

    def measure_inference_time(self, model1, model2, test_loader, device):
        start_time = time.time()

        model1.eval()
        model2.eval()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)
                _ = outputs1 + outputs2

        self.inference_time = time.time() - start_time

    def end_training(self):
        training_time = time.time() - self.start_time
        max_gpu_mem = max_memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB

        df = pd.DataFrame(self.metrics)
        df.to_csv('training_history.csv', index=False)

        with open('performance_metrics.txt', 'w') as f:
            f.write(f"Total training time: {training_time:.2f} seconds\n")
            f.write(f"Peak GPU memory usage: {max_gpu_mem:.2f} GB\n")
            f.write(f"Inference time: {self.inference_time:.2f} seconds" if hasattr(self, 'inference_time') else "")
