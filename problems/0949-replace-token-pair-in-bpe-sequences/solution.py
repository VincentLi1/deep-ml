def replace_pair(sequences, pair, new_id):
    # Your code here
    ans_arr = []
    for seq in sequences:
        ans = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and pair == (seq[i], seq[i + 1]):
                ans.append(new_id)
                i += 2
            else:
                ans.append(seq[i])
                i += 1
        ans_arr.append(ans)
    return ans_arr