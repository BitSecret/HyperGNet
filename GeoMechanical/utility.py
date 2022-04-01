def get_all_representation_of_shape(shape):
    results = []
    length = len(shape)
    for i in range(length):
        result = ""
        for j in range(length):
            result += shape[(i + j) % length]
        results.append(result)
    return results


def pre_parse(fl):
    if fl[0] == "PointOn":
        if fl[2][0] == "Line":
            fl[0] = "PointOnLine"
        elif fl[2][0] == "Arc":
            fl[0] = "PointOnArc"
        else:
            fl[0] = "PointOnCircle"
    elif fl[0] == "Disjoint":
        if fl[1][0] == "Line":
            fl[0] = "DisjointLineCircle"
        else:
            fl[0] = "DisjointCircleCircle"
    elif fl[0] == "Tangent":
        if fl[2][0] == "Line":
            fl[0] = "TangentLineCircle"
        else:
            fl[0] = "TangentCircleCircle"
    elif fl[0] == "Intersect":
        if fl[2][0] == "Line":
            fl[0] = "IntersectLineLine"
        elif fl[3][0] == "Line":
            fl[0] = "IntersectLineCircle"
        else:
            fl[0] = "IntersectCircleCircle"
    elif fl[0] == "Height":
        if fl[2][0] == "Triangle":
            fl[0] = "HeightOfTriangle"
        else:
            fl[0] = "HeightOfTrapezoid"

    return fl
