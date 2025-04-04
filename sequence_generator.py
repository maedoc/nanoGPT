import numpy as np

def generate_sequence(length):
    """Generate a sequence of 4 random numbers between 0 and 9, repeated until the desired length."""
    base_sequence = np.random.randint(0, 10, size=4)
    full_sequence = np.tile(base_sequence, (length // 4 + 1))[:length]
    return full_sequence

def generate_batch(batch_size, num_tokens):
    """Generate a batch of sequences."""
    batch = [generate_sequence(num_tokens) for _ in range(batch_size)]
    return np.array(batch)

# Example usage
if __name__ == "__main__":
    print(generate_batch(5, 16))
