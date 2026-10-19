def most_frequent_pair(sequences):
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """
    if not sequences or max(len(s) for s in sequences) < 2:
        return None

    counts = {}
    curr_best_pair = (-1, -1, '', '')
    curr_max = 0
    for i, seq in enumerate(sequences):
        for j in range(len(seq) - 1):
            pair = (i, j, seq[j], seq[j + 1])
            counts[pair] = counts.get(pair, 0) + 1

            if counts[pair] > curr_max:
                curr_max = counts[pair]
                curr_best_pair = pair
            elif counts[pair] == curr_max:
                if curr_best_pair[0] > pair[0]:
                    curr_best_pair = pair
                if curr_best_pair[0] == pair[0] and curr_best_pair[1] > pair[1]:
                    curr_best_pair = pair
    return curr_best_pair[2:]

    
    