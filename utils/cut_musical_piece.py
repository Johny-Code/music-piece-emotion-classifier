def cut_musical_piece(x, sr, time, division="middle"):
    if (division == "middle"):
        middle = int(x.shape[0] / 2)
        lower = int(middle - sr * time / 2)
        upper = int(middle + sr * time / 2)
        return x[lower:upper]
    elif(division == "beginning"):
        return x[0:int(sr * time)]
    elif(division == "end"):
        return x[-int(sr * time):]