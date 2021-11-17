# from typing import Tuple
# import numpy as np
# import torch.nn as nn

# from replaybuffer import ReplayBuffer

# DELTA = 1e-3


# class GaussianEnv:
#     def __init__(self, a: float = 100.0, b: float = 0.0, c: float = 1.0) -> None:
#         self.a = a
#         self.b = b
#         self.c = c

#     def reset(self) -> None:
#         self.t = 0
#         obs = 0.0
#         return obs

#     def step(self, action: float) -> float:
#         self.t += 1
#         f = lambda x: self.a * np.exp(-(x - self.b) / (2 * self.c ** 2))
#         obs = f(action)
#         return obs


# class RNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # self.rnn = nn.GRUCell(input_size=10, hidden_size=2)
#         self.rnn = nn.GRU()
#         self.linear = nn.Linear(in_features=2, out_features=1)
    
#     def forward(x):
#         pass


# class SimpleAgent:
#     def __init__(self, search_range: Tuple[float, float] = (-1e2, 1e2)) -> None:
#         self.buffer = ReplayBuffer(max_size=100)
#         self.buffer.initialize_buffer("observations")
#         self.buffer.initialize_buffer("actions")
#         self.buffer.initialize_buffer("ranges")

#         # self.search_range = search_range
#         # self.t = 0

#         self.W_1 = np.random.rand()
#         self.b_1 = 0.0


#     def reward(self):
#         self.buffer.sample(n=10)



#     def step(self, obs: float) -> float:


#     # def step(self, obs: float) -> float:
#     #     previous = self.buffer.previous(1)
#     #     if self.t == 0:
#     #         low, high = self.search_range
#     #     else:
#     #         low, high = previous["ranges"][0]

#     #     reward = previous["rewards"][0]
#     #     action = low + (high - low) / 2
#     #     rng = self.get_range(low, high, action)
#     #     print(f"rng = {rng}")

#     #     self.buffer.store(observations=obs, actions=action, ranges=rng)
#     #     self.t += 1

#     #     return action

#     # @staticmethod
#     # def get_range(low, high, candidate):
#     #     midpoint = low + (high - low) / 2
#     #     if low <= candidate < midpoint:
#     #         return (low, midpoint)
#     #     else:
#     #         return (midpoint, high)


# def main():
#     env = GaussianEnv()
#     agent = SimpleAgent()

#     obs = env.reset()

#     prev_action = np.inf
#     while True:
#         action = agent.step(obs=obs)
#         obs = env.step(action=action)

#         print(f"obs = {obs}")
#         print(f"action = {action}")

#         if abs(action - prev_action) < DELTA:
#             break

#         if env.t == 5:
#             break


# if __name__ == "__main__":
#     main()
