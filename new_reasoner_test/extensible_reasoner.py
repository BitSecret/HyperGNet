from kanren import Relation, run, vars, facts

pdc_dfl = ["Triangle", "LEQ", "AEQ"]
theo_dfl = [
    {"name": "el_to_ea",
     "premise": [("Triangle", "ABC"), ("LEQ", "AB", "AC")],
     "conclusion": [("AEQ", "ABC", "BCA")]},
    {"name": "ea_to_el",
     "premise": [("Triangle", "ABC"), ("AEQ", "ABC", "BCA")],
     "conclusion": [("LEQ", "AB", "AC")]}
]

test_fl_json = {
    "name": "attr_of_el",
    "premise": {
        "entity_relation": ["Triangle(ABC)", "Line(AB)", "Line(AC)", "Angle(ABC)", "Angle(BCA)"],
        "algebraic_relation": ["Equal(Length(AB),Length(AC))"]
    },
    "conclusion": {
        "entity_relation": ["IsoscelesTriangle(ABC)"],
        "algebraic_relation": ["Equal(Measure(ABC),Measure(BCA))"]
    }
}

conditions = {"Triangle": [("A", "B", "C"), ("A", "C", "D"), ("E", "F", "G")],
              "LEQ": [("A", "B", "A", "C")],
              "AEQ": [("E", "F", "G", "F", "G", "E")]}


class Problem:

    def __init__(self):
        self.relation = {}
        self.theo = {}

    def predicate_define(self, dfl):  # 定义谓词
        for predicate in dfl:
            self.relation[predicate] = Relation(predicate)

    def theorem_define(self, dfl):  # 定义定理
        for d in dfl:
            var = {}  # 暂时储存变量

            premise = []  # 输入run函数的premise
            for p in d["premise"]:
                r_p = self.relation[p[0]]
                var_list = ""
                for i in p[1:len(p)]:
                    var_list += i
                var_list = list(var_list)
                for i in range(len(var_list)):
                    if var_list[i] not in var:
                        var[var_list[i]], = vars(1)
                    var_list[i] = var[var_list[i]]
                premise.append(r_p(tuple(var_list)))

            conclusion = []  # 输入run函数的x
            for c in d["conclusion"]:
                var_list = ""
                for i in c[1:len(c)]:
                    var_list += i
                var_list = list(var_list)
                for i in range(len(var_list)):
                    if var_list[i] not in var:
                        var[var_list[i]], = vars(1)
                    var_list[i] = var[var_list[i]]
                conclusion.append([c[0], tuple(var_list)])

            self.theo[d["name"]] = [tuple(premise), conclusion]

    def add_condition(self, data):  # 添加条件
        for predicate in data:
            for item in data[predicate]:
                facts(self.relation[predicate], item)

    def apply_theorem(self, theorem):
        premise, conclusions = self.theo[theorem]

        for conclusion in conclusions:
            results = run(0, conclusion[1], premise)
            for result in results:
                facts(self.relation[conclusion[0]], result)

    def show(self):
        for pre in self.relation:
            print(pre, end=": ")
            print(self.relation[pre].facts)


def main():
    problem = Problem()
    problem.predicate_define(pdc_dfl)   # 定义谓词
    problem.theorem_define(theo_dfl)    # 定义定理
    problem.add_condition(conditions)    # 添加题目条件

    problem.show()
    print()

    problem.apply_theorem("ea_to_el")
    problem.show()
    print()

    problem.apply_theorem("el_to_ea")
    problem.show()


if __name__ == '__main__':
    main()
