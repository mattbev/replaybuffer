# ReplayBuffer
The simple buffer for experience replay. Built for uses in reinforcement learning, computer vision, and other applications where temporal information matters. 


## Installation

### Using `pip`
Clone this repository:
```bash
git clone git@github.com:mattbev/replaybuffer.git
```
Install using `pip`:
```bash
pip install -e <path_to_repo_base_directory>
```

### From source
This package only uses `numpy` and python built-ins -- just clone the repo and `pip install numpy` to use it in your project. 

## Usage
This package is meant to be used for experience replay with any data types. To do that, first initialize the buffer:
```python
from replaybuffer import ReplayBuffer

buffer = ReplayBuffer(max_size=100)
buffer.initialize_buffer("observations")
buffer.initialize_buffer("actions")
buffer.initialize_buffer("rewards")
...
```

Then, within your project, you can store data:
```python
...
 env = <some environment, e.g., OpenAI Gym>
 obs, reward = env.step(action)
 
 buffer.store(
     observations=obs, # image
     actions=action, # vector
     rewards=reward # float
 )
 # or 
 # buffer.store(
 #    **{
 #        "observations": obs, # image
 #        "actions": action, # vector
 #        "rewards": reward # float
 #    }
 #)
 ...
```

And then retreive it:
```python
...
replay = buffer.sample(n=10)
<use the replay to revisit past observations, for example>
...
```

A simple example can be found at [samples/basics.py](samples/basics.py).

