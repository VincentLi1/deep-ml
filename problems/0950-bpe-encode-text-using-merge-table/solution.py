def bpe_encode(text, token_to_id, merges):
    # Your code here
    ids = [token_to_id[tok] for tok in text]
    ans = []
    i = 0
    while i < len(ids):
        j = i + 1
        curr_tok = ids[i]
        while j < len(ids) and (curr_tok, ids[j]) in merges:
            curr_tok = merges[(curr_tok, ids[j])]
            j += 1

        ans.append(curr_tok)
        i = j

    return ans