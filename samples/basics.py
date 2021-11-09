from replaybuffer import ReplayBuffer


def main():
    buffer = ReplayBuffer(max_size=100)
    buffer.initialize_buffer("stream_1")
    buffer.initialize_buffer("stream_2")

    # for i in range(1000):
