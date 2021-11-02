def pad_sequence(seq, target_length, filler=None):
    filler_count = 0

    if len(seq) < target_length:
        filler_count = target_length - len(seq)

    return seq + [filler] * filler_count
