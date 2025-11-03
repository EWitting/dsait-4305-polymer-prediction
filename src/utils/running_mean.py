from collections import deque

class RunningMean:
    def __init__(self, maxlen=5):
        self.values = deque(maxlen=maxlen)

    def update(self, value):
        self.values.append(value)

    @property
    def mean(self):
        return sum(self.values)/len(self.values)