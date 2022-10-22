import time
import numpy as np

class LatencyProfiler:
    def __init__(self) -> None:
        self.execution_times = []
        self.avg_execution_times = []
        self.current_start_time = 0
        self.current_end_time = 0

    def start_run(self):
        self.execution_times.append([])

    def end_run(self):
        self.avg_execution_times.append(np.mean(self.execution_times[-1]))

    def start(self):
        self.current_start_time = time.time()

    def end(self):
        self.current_end_time = time.time()

        self.execution_times[-1].append(self.current_end_time - self.current_start_time)

    def compute_metrics(self):
        mean = np.mean(self.avg_execution_times)
        std = np.std(self.avg_execution_times)

        return mean, std, self.avg_execution_times