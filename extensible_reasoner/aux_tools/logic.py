def run(relations, ops):
    """
    Reasoning.
    :param relations: list of <function> from <Relation>'s __call__()
    :param ops: list of <str>, operation between relations, 2 forms: "and" and "not"
    :return: reasoning result: (ids, items, variables) or None
    """
    r1_ids, r1_items, r1_vars = relations[0]()
    i = 1
    while i < len(relations):
        r1_ids, r1_items, r1_vars = combine((r1_ids, r1_items, r1_vars), relations[i](), ops[i - 1])

        if len(r1_ids) == 0:
            return None, None, None
        i += 1

    return r1_ids, r1_items, r1_vars


def combine(r1, r2, op):
    """Combine r1 and r2 to a new relation r."""
    r1_ids, r1_items, r1_vars = r1    # r1,r2的vars都是按0,1,...排好的
    r2_ids, r2_items, r2_vars = r2
    r_ids, r_items = [], []

    r_vars = tuple(set(r1_vars) | set(r2_vars))  # r1和r2变量的并集  这个会按变量的先后顺序0,1,...排好
    union = list(set(r1_vars) & set(r2_vars))    # 共有变量
    for i in range(len(union)):
        union[i] = (r1_vars.index(union[i]), r2_vars.index(union[i]))  # 转化为索引
    difference = list(set(r2_vars) - set(r1_vars))    # 变量的差集: r2-r1
    for i in range(len(difference)):
        difference[i] = r2_vars.index(difference[i])  # 转化为索引

    for i in range(len(r1_items)):
        for j in range(len(r2_items)):
            r1_data = r1_items[i]
            r2_data = r2_items[j]
            if constrained(r1_data, r2_data, union, op):
                item = list(r1_data)
                for dif in difference:
                    item.append(r2_data[dif])
                r_items.append(tuple(item))
                r_ids.append(tuple(set(list(r1_ids[i]) + list(r2_ids[j]))))

    return r_ids, r_items, r_vars


def constrained(r1_data, r2_data, union, op):
    """Judge whether r1_data and r2_data satisfy the operation at the corresponding position."""
    for u in union:
        if r1_data[u[0]] != r2_data[u[1]]:    # 存在对应item不一致
            return not op == "and"
    return op == "and"    # 对应item一致
