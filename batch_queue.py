from collections import defaultdict
from threading import Lock, Thread
import time

class BatchQueue:
    def __init__(self, training_queue, batch_size=1024, flush_interval=5):
        """
        Initializes the BatchQueue.

        Args:
            training_queue (multiprocessing.Queue): The queue to send batched data to.
            batch_size (int): The number of data points to accumulate before sending.
            flush_interval (int): Time interval in seconds to flush remaining data.
        """
        self.training_queue = training_queue
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffers = defaultdict(list)  # Buffers per model_type
        self.lock = Lock()
        self.running = True
        self.flush_thread = Thread(target=self._flush_periodically, daemon=True)
        self.flush_thread.start()

    def add_data_point(self, port, data_buffer, model_type, log=True):
        """
        Adds a data point to the buffer. If the buffer reaches the batch size, sends it to the queue.

        Args:
            port (int): The port number associated with the data.
            data_buffer (list): The list of data points to add.
            model_type (str): The type of model ("Battle_Model" or "Planning_Model").
            log (bool): Whether to enable logging.
        """
        with self.lock:
            self.buffers[model_type].extend(data_buffer)
            current_length = len(self.buffers[model_type])
            if current_length >= self.batch_size:
                batch = self.buffers[model_type][:self.batch_size]
                self.buffers[model_type] = self.buffers[model_type][self.batch_size:]
                self.training_queue.put((port, batch, model_type, log))
                print(f"BatchQueue: Sent batch of {self.batch_size} for {model_type} from port {port}.")

    def _flush_periodically(self):
        """
        Periodically flushes the buffers, sending whatever data has been accumulated.
        """
        while self.running:
            time.sleep(self.flush_interval)
            with self.lock:
                for model_type, buffer in list(self.buffers.items()):
                    if buffer:
                        # For periodic flush, set port to None or a default value
                        self.training_queue.put((None, buffer.copy(), model_type, True))
                        print(f"BatchQueue: Flushed {len(buffer)} data points for {model_type}.")
                        self.buffers[model_type].clear()

    def stop(self):
        """
        Stops the periodic flushing and sends any remaining data.
        """
        self.running = False
        self.flush_thread.join()
        with self.lock:
            for model_type, buffer in list(self.buffers.items()):
                if buffer:
                    self.training_queue.put((None, buffer.copy(), model_type, True))
                    print(f"BatchQueue: Final flush of {len(buffer)} data points for {model_type}.")
                    self.buffers[model_type].clear()
