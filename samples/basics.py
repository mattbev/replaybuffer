from replaybuffer import ReplayBuffer
from replaybuffer.utils import remove_nones


def main():
    buffer = ReplayBuffer(max_size=100)
    buffer.initialize_buffer("stream_1")
    buffer.initialize_buffer("stream_2")

    print("Initial state")
    print(buffer.previous(n=10))
    print()

    buffer.store(stream_1=1, stream_2="some other data")

    print("After adding one element")
    print(buffer.previous(n=10))
    print()

    print("Cleaned up")
    prev = buffer.previous(n=10)
    s1, s2 = remove_nones(*prev.values())
    print(s1)
    print(s2)
    print()

    print("Sampling")
    for i in range(100):
        buffer.store(stream_1=i, stream_2=i + 1)

    print(buffer.sample(n=10))
    print()

    print("Taking")
    print(buffer.take(range(0, 5)))
    print(buffer[0:5])


if __name__ == "__main__":
    main()
