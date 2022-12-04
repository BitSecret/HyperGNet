from kanren import Relation, run, vars, var, facts


def main():
    a, b, c, x, y, z = vars(6)

    tri_data = [("A", "B", "C"), ("A", "C", "D"), ("E", "F", "G")]
    l_eq_data = [("A", "B", "A", "C"), ("E", "F", "E", "G")]

    tri = Relation()  # 三角形
    for data in tri_data:
        facts(tri, data)
    print(tri.facts)
    l_eq = Relation()  # 边相等
    for data in l_eq_data:
        facts(l_eq, data)
    print(l_eq.facts)
    a_eq = Relation()  # 角相等

    print()
    results = run(0, (a, b, c, b, c, a), (tri((a, b, c)), l_eq((a, b, a, c))))
    for result in results:
        facts(a_eq, result)
    print(a_eq.facts)


def test1():
    a = var("a")
    b = var("b")
    c = var("c")
    d = var("d")
    x = var("x")
    y = var("y")
    z = var("z")

    tri_data = [("A", "B", "C"),
                ("B", "C", "D"),
                ("B", "C", "E"),
                ("D", "E", "F")]

    tri = Relation()  # 三角形
    for data in tri_data:
        facts(tri, data)

    # func = tri((x, y, z))
    # func = tri(("A", "B", z))
    # func = tri(("C", y, z))
    # for i in func({}):
    #     pass

    # results = run(0, (x, y, z), tuple([tri((x, y, z))]))
    results = run(0, (a, b, c, d), (tri((a, b, c)), tri((b, c, d))))
    # print(results)


if __name__ == '__main__':
    test1()
