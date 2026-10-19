from collections import defaultdict

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.

    Args:
        corpus: Dictionary mapping space-separated token sequences to frequencies.
        num_merges: Number of merge operations to perform.

    Returns:
        List of tuples representing merged token pairs.
    """
    corpus = {
        tuple(word.split()): freq
        for word, freq in corpus.items()
    }

    merges = []

    for _ in range(num_merges):
        # Count frequencies of adjacent token pairs
        pair_counts = defaultdict(int)

        for tokens, freq in corpus.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq

        # Stop if no pairs remain
        if not pair_counts:
            break

        # Find the most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)

        # Merge the selected pair throughout the corpus
        new_corpus = {}

        for tokens, freq in corpus.items():
            merged_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    merged_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1

            merged_tokens = tuple(merged_tokens)
            new_corpus[merged_tokens] = (
                new_corpus.get(merged_tokens, 0) + freq
            )

        corpus = new_corpus

    return merges