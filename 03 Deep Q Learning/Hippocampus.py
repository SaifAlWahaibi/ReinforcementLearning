
import numpy as np


class Memory:
    def __init__(self, memory_size, input_shape):
        self.mem_dim = memory_size
        self.inp_dim = input_shape

        self.ctr = 0

        self.sta = np.zeros((self.mem_dim, self.inp_dim), dtype=np.float32)
        self.act = np.zeros(self.mem_dim, dtype=np.int64)
        self.rwd = np.zeros(self.mem_dim, dtype=np.float32)
        self.fut_sta = np.zeros((self.mem_dim, self.inp_dim), dtype=np.float32)
        self.ter = np.zeros(self.mem_dim, dtype=np.bool)

    def save(self, state, action, reward, future_state, terminal):
        idx = self.ctr % self.mem_dim

        self.sta[idx] = state
        self.act[idx] = action
        self.rwd[idx] = reward
        self.fut_sta[idx] = future_state
        self.ter[idx] = terminal

        self.ctr += 1

    def sampler(self, batch_size):
        mem_idx = min(self.ctr, self.mem_dim)
        bat_idx = np.random.choice(mem_idx, batch_size, replace=False)

        state = self.sta[bat_idx]
        action = self.act[bat_idx]
        reward = self.rwd[bat_idx]
        future_state = self.fut_sta[bat_idx]
        terminal = self.ter[bat_idx]

        return state, action, reward, future_state, terminal
