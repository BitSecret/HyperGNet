def get_all_representation_of_shape(shape):
    results = []
    length = len(shape)
    for i in range(length):
        result = ""
        for j in range(length):
            result += shape[(i + j) % length]
        results.append(result)
    return results
