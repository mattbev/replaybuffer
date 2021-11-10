from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBuffer:
    max_size: int = np.inf

    def __post_init__(self):
        self.buffers = set()
        self.ptr, self.size = 0, 0


    def initialize_buffer(self, buffer_name: str):
        setattr(self, buffer_name, np.empty(self.max_size, dtype=object))
        self.buffers.add(buffer_name)


    def store(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k)[self.ptr] = v

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    

    def sample(self, batch_size: int = 10):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self[idx]


    def previous(self, batch_size: int = 10):
        assert batch_size <= self.max_size
        return self.take(range(self.ptr-batch_size, self.ptr), mode='wrap')


    def take(self, *args, **kwargs):
        return {k: getattr(self, k).take(*args, **kwargs) for k in self.buffers}

    
    def set_parameter(self, parameter, idx, val):
        self.__dict__[parameter][idx] = val


    def get_parameter(self, parameter, idx):
        return self.__dict__[parameter][idx]


    @staticmethod
    def remove_nones(*arrays):
        l = []
        for array in arrays:
            l.append([i for i in array if i is not None])
        return tuple(l)


    def __getitem__(self, idx):
        return {k: getattr(self, k)[idx] for k in self.buffers}


    def __len__(self):
        return self.size



if __name__ == '__main__':
    buffer = ReplayBuffer(max_size=10)
    buffer.initialize_buffer('X_W_chain')

    for i in range(10):
        buffer.store(
            X_W_chain = np.array([i])
        )

    print(buffer.get_parameter("X_W_chain", range(0, 4)))
    print(buffer.previous(3))
    # buffer.set_parameter('X_W_chain', range(0, 2), np.array([5, 6]))
    # print(buffer.get_parameter("X_W_chain", range(0, 3)))

    # print(buffer.previous(12)['X_W'].shape)
    # print(buffer.previous(4)['X_W'])
