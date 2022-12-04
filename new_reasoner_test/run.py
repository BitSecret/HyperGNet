class Relation:

    def __init__(self, name="default"):
        self.facts = []
        self.name = name

    def add_fact(self, item):
        if item not in self.facts:
            self.facts.append(item)

    def __call__(self, variables):
        def get_data():
            return self.facts, variables
        return get_data

    def __str__(self):
        return self.name + ": " + str(self.facts)


def run(ls):
    """
    :param ls: tuple of Logic
    :return: Logic
    """
    logic1 = ls[0]
    i = 1
    while i < len(ls):
        logic2 = ls[i]
        logic1 = reasoning(logic1, logic2)
        i += 1

    return logic1()


def reasoning(logic1, logic2):
    l1_facts, l1_vars = logic1()
    l2_facts, l2_vars = logic2()
    variables = tuple(set(l1_vars) | set(l2_vars))  # 这是变量
    union = list(set(l1_vars) & set(l2_vars))
    for i in range(len(union)):
        union[i] = (l1_vars.index(union[i]), l2_vars.index(union[i]))  # 转化为索引
    difference = list(set(l2_vars) - set(l1_vars))
    for i in range(len(difference)):
        difference[i] = l2_vars.index(difference[i])  # 转化为索引

    uniform = []  # 符合条件的数据对
    for l1_data in l1_facts:
        for l2_data in l2_facts:
            join_data = True
            for u in union:
                if l1_data[u[0]] != l2_data[u[1]]:
                    join_data = False
                    break
            if join_data:
                item = list(l1_data)
                for i in difference:
                    item.append(l2_data[i])
                uniform.append(tuple(item))

    new_relation = Relation()
    for item in uniform:
        new_relation.add_fact(item)

    return new_relation(variables)


def main():
    tri = Relation("tri")
    tri.add_fact(("A", "B", "C"))
    tri.add_fact(("D", "E", "F"))
    line_eq = Relation("line_eq")
    line_eq.add_fact(("A", "B", "A", "C"))
    angle_eq = Relation("angle_eq")
    angle_eq.add_fact(("D", "E", "F", "E", "F", "D"))

    print(tri)
    print(line_eq)
    print(angle_eq)
    print()

    ea_to_el_pre = (tri((0, 1, 2)), angle_eq((0, 1, 2, 1, 2, 0)))  # 注：数字可看作变量
    facts, variables = run(ea_to_el_pre)     # 模拟定理应用，实际更复杂
    for r in facts:
        line_eq.add_fact((r[0], r[1], r[0], r[2]))

    print(tri)
    print(line_eq)
    print(angle_eq)
    print()

    el_to_ea_pre = (tri((0, 1, 2)), line_eq((0, 1, 0, 2)))  # 注：数字可看作变量
    facts, variables = run(el_to_ea_pre)     # 模拟定理应用，实际更复杂
    for r in facts:
        angle_eq.add_fact((r[0], r[1], r[2], r[1], r[2], r[0]))

    print(tri)
    print(line_eq)
    print(angle_eq)
    print()


if __name__ == '__main__':
    main()
