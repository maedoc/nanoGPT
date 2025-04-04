import numpy as np

def generate_sequence(length):
    """Generate a sequence of random length (between 2 and 5) random numbers between 0 and 9, repeated until the desired length."""
    base_length = np.random.randint(2, 6)
    base_sequence = np.random.randint(0, 10, size=base_length)
    full_sequence = np.tile(base_sequence, (length // 4 + 1))[:length]
    return full_sequence

def generate_batch(batch_size, num_tokens):
    """Generate a batch of sequences with fixed length."""
    batch = []
    for _ in range(batch_size):
        seq = generate_sequence(num_tokens)
        if len(seq) < num_tokens:
            seq = np.pad(seq, (0, num_tokens - len(seq)), 'constant')
        else:
            seq = seq[:num_tokens]
        batch.append(seq)
    return np.array(batch)
    return np.array(batch)

# Example usage
if __name__ == "__main__":
    print(generate_batch(5, 16))
