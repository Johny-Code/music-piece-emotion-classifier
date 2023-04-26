def cut_musical_piece(x, sr, time):
    middle = int(x.shape[0] / 2)
    lower = int(middle - sr * time / 2)
    upper = int(middle + sr * time / 2)
    return x[lower:upper]
