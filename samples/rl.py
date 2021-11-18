# from typing import Tuple
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# from replaybuffer import ReplayBuffer

# DELTA = 1e-3


# # class GaussianEnv:
# #     def __init__(self, a: float = 100.0, b: float = 0.0, c: float = 1.0) -> None:
# #         self.a = a
# #         self.b = b
# #         self.c = c

# #     def reset(self) -> None:
# #         self.t = 0
# #         obs = 0.0
# #         return obs

# #     def step(self, action: float) -> float:
# #         self.t += 1
# #         f = lambda x: self.a * np.exp(-(x - self.b) / (2 * self.c ** 2))
# #         obs = f(action)
# #         return obs

# class GridWorldEnv:
#     def __init__(self, size: Tuple[int, int] = (10, 10)) -> None:
#         self.size = size

#     def reset(self):
#         self.goal_coords = (
#             np.random.randint(low=0, high=self.size[0]),
#             np.random.randint(low=0, high=self.size[1])
#         )
#         self.agent_coords = (
#             np.random.randint(low=0, high=self.size[0]),
#             np.random.randint(low=0, high=self.size[1])
#         )
#         obs = self.make_obs()
        
#         return obs

#     def move_left(self):
#         self.agent_coords = (
#             self.agent_coords[0],
#             self.agent_coords[1] - 1
#         )
    
#     def move_right(self):
#         self.agent_coords = (
#             self.agent_coords[0],
#             self.agent_coords[1] + 1
#         )

#     def move_up(self):
#         self.agent_coords = (
#             self.agent_coords[0] -1,
#             self.agent_coords[1]
#         )

#     def move_down(self):
#         self.agent_coords = (
#             self.agent_coords[0] + 1,
#             self.agent_coords[1]
#         )

#     def make_obs(self):
#         obs = np.full((*self.size, 3), fill_value=0)
#         obs[self.goal_coords] = (255, 200, 0) # gold
#         obs[self.agent_coords] = (255, 255, 255) # black
#         # plt.imshow(obs)

#         return obs

#     def step(self, action):
#         if action == "l":
#             self.move_left()
#         elif action == "r":
#             self.move_right()
#         elif action == "u":
#             self.move_up()
#         elif action == "d":
#             self.move_down()
#         else:
#             raise Exception(f"Invalid action command: {action}")

#         reward = 1.0 / (1.0 + np.linalg.norm(self.goal_coords - self.agent_coords))
#         obs = self.make_obs()
#         done = self.agent_coords == self.goal_coords

#         return reward, obs, done



# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv = nn.Conv2d(
#             in_channels=3, 
#             out_channels=10, 
#             kernel_size=3
#         )
#         self.flatten = nn.Flatten(start_dim=1)
#         self.rnn = nn.GRUCell(
#             input_size=10,
#             hidden_size=1
#         )
#         self.linear = nn.Linear(in_features=2, out_features=1)
    
#     def forward(self, x: torch.Tensor):
#         # x is shape (B, L, H, W, C)
#         L = x.size(1)
#         outputs = torch.Tensor(L)
#         hx = torch.rand(L, )
#         for l in range(L):
#             x = self.conv(x)
#             x = self.flatten(x)
#             x, hx = self.rnn(x)
#             x = self.linear(x)
        
#         return x


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



#     # def step(self, obs: float) -> float:


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
#     env = GridWorldEnv()
#     agent = SimpleAgent()

#     obs = env.reset()

#     # prev_action = np.inf
#     # while True:
#     #     action = agent.step(obs=obs)
#     #     obs = env.step(action=action)

#     #     print(f"obs = {obs}")
#     #     print(f"action = {action}")

#     #     if abs(action - prev_action) < DELTA:
#     #         break

#     #     if env.t == 5:
#     #         break


# if __name__ == "__main__":
#     main()
