import tensorflow as tf
import random

# TODO: Add method comments
class Memory:
    """
    Defining the memory class
    """
    def __init__(self, max_mem):
        self._max_mem = max_mem
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_mem:
            self._samples.pop(0)
    
    def sample(self, sample_count):
        if sample_count > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, sample_count)
