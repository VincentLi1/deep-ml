def welford_stats(data, ddof=1):
    # data: iterable of numbers
    # return (count, mean, variance)
    if not data:
        return (0, 0.0, 0.0)
    
    count = 0
    running_sum = 0
    running_sum_squares = 0
    first_datum = None
    for datum in data:
        count += 1

        # Recenter
        if first_datum is None:
            first_datum = datum
            datum = 0
        else:
            datum -= first_datum

        running_sum += datum
        running_sum_squares += datum ** 2

    mean = running_sum / count + first_datum
    if count - ddof <= 0:
        variance = 0.0
    else:
        variance = 1 / (count - ddof) * (running_sum_squares - count * (running_sum / count) ** 2)
    
    return (count, 
    mean, variance)

    '''
    var = 1 / (count - ddof) * [sum (x_i - x_bar) ** 2]
    = 1 / (count - ddof) * [ sum x_i ** 2 - 2 sum x_i x_bar + count * x_bar ** 2 ]
    = 1 / (count - ddof) * [ sum x_i ** 2 - count x_bar ** 2 ]
    '''
