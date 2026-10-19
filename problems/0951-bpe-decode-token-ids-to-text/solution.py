def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    ans = []
    for id_ in ids:
        tok = vocab[id_]
        if tok[0] == 'G':
            tok = ' ' + tok[1:]
        ans.append(tok)
    return ''.join(ans)