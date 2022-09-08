import json
import math
import random
import string
import time
random.seed(1999)


class Generator:

    def __init__(self, init_count):
        self.problem = {}
        self.count = init_count

    def generate_problem_using_template_1(self, problem_number):
        i = 0
        while i < problem_number:
            problem_id = self.count
            annotation = "generated_{}".format(time.strftime("%Y-%m-%d", time.localtime()))
            source = "generating_from_template_1"
            problem_level = 1
            problem_text_cn = "在直角三角形{}中，{}={}，{}={}，求{}的长度。"
            problem_text_en = "In the right triangle {}, {}={}, {}={}, find the length of {}."
            problem_img = "template_1.png"
            problem_answer = []
            construction_fls = ["Shape({})"]
            text_fls = ["RightTriangle({})", "Equal(Length(Line({})),{})", "Equal(Length(Line({})),{})"]
            image_fls = []
            target_fls = ["Find(Length(Line({})))"]
            theorem_seqs = [22]
            completeness = "True"

            selection = random.randint(1, 2)
            length_line1 = random.randint(1, 100)
            length_line2 = random.randint(1, length_line1)    # line2 < line1
            shape_name = Generator._random_letters_upper(3)
            if selection == 1:
                line1_name = shape_name[0:2]
                line2_name = shape_name[1:3]
                line3_name = shape_name[2] + shape_name[0]
                answer = math.sqrt(math.pow(length_line1, 2) + math.pow(length_line2, 2))
            else:
                if length_line1 == length_line2:    # 构不成三角形
                    continue
                line1_name = shape_name[2] + shape_name[0]
                line2_name = shape_name[0:2]
                line3_name = shape_name[1:3]
                answer = math.sqrt(math.pow(length_line1, 2) - math.pow(length_line2, 2))

            problem_text_cn = problem_text_cn.format(shape_name, line1_name, length_line1,
                                                     line2_name, length_line2, line3_name)
            problem_text_en = problem_text_en.format(shape_name, line1_name, length_line1,
                                                     line2_name, length_line2, line3_name)
            problem_answer.append("{:.3f}".format(answer))
            construction_fls[0] = construction_fls[0].format(shape_name)
            text_fls[0] = text_fls[0].format(shape_name)
            text_fls[1] = text_fls[1].format(line1_name, length_line1)
            text_fls[2] = text_fls[2].format(line2_name, length_line2)
            target_fls[0] = target_fls[0].format(line3_name)

            data_unit = {
                "problem_id": problem_id,
                "annotation": annotation,
                "source": source,
                "problem_level": problem_level,
                "problem_text_cn": problem_text_cn,
                "problem_text_en": problem_text_en,
                "problem_img": problem_img,
                "problem_answer": problem_answer,
                "construction_fls": construction_fls,
                "text_fls": text_fls,
                "image_fls": image_fls,
                "target_fls": target_fls,
                "theorem_seqs": theorem_seqs,
                "completeness": completeness
            }

            self.add(data_unit)
            i += 1

    def generate_problem_using_template_2(self, problem_number):
        i = 0
        while i < problem_number:
            problem_id = self.count
            annotation = "generated_{}".format(time.strftime("%Y-%m-%d", time.localtime()))
            source = "generating_from_template_2"
            problem_level = 1
            problem_text_cn = "三角形{}是直角三角形，求{}的值。"
            problem_text_en = "The triangle {} is a right triangle. Find the value of {}."
            problem_img = "template_2.png"
            problem_answer = []
            construction_fls = ["Shape({})"]
            text_fls = ["RightTriangle({})"]
            image_fls = ["Equal(Length(Line({})),{})", "Equal(Length(Line({})),{})", "Equal(Length(Line({})),{})"]
            target_fls = ["Find({})"]
            theorem_seqs = [22]
            completeness = "True"

            selection = random.randint(1, 2)
            length_line1 = random.randint(1, 100)
            length_line2 = random.randint(1, length_line1)    # line2 < line1
            shape_name = Generator._random_letters_upper(3)
            variable_name = Generator._random_letters_lower(1)
            if selection == 1:
                line1_name = shape_name[0:2]
                line2_name = shape_name[1:3]
                line3_name = shape_name[2] + shape_name[0]
                answer = math.sqrt(math.pow(length_line1, 2) + math.pow(length_line2, 2))
            else:
                if length_line1 == length_line2:    # 构不成三角形
                    continue
                line1_name = shape_name[2] + shape_name[0]
                line2_name = shape_name[0:2]
                line3_name = shape_name[1:3]
                answer = math.sqrt(math.pow(length_line1, 2) - math.pow(length_line2, 2))

            problem_text_cn = problem_text_cn.format(shape_name, variable_name)
            problem_text_en = problem_text_en.format(shape_name, variable_name)
            problem_answer.append("{:.3f}".format(answer))
            construction_fls[0] = construction_fls[0].format(shape_name)
            text_fls[0] = text_fls[0].format(shape_name)
            image_fls[0] = image_fls[0].format(line1_name, length_line1)
            image_fls[1] = image_fls[1].format(line2_name, length_line2)
            image_fls[2] = image_fls[2].format(line3_name, variable_name)
            target_fls[0] = target_fls[0].format(variable_name)

            data_unit = {
                "problem_id": problem_id,
                "annotation": annotation,
                "source": source,
                "problem_level": problem_level,
                "problem_text_cn": problem_text_cn,
                "problem_text_en": problem_text_en,
                "problem_img": problem_img,
                "problem_answer": problem_answer,
                "construction_fls": construction_fls,
                "text_fls": text_fls,
                "image_fls": image_fls,
                "target_fls": target_fls,
                "theorem_seqs": theorem_seqs,
                "completeness": completeness
            }

            self.add(data_unit)
            i += 1

    def add(self, problem_unit):
        self.problem[str(self.count)] = problem_unit
        self.count += 1

    def save(self):
        with open("../data/generated_data/new_generated_{}.json".format(int(time.time())), "w", encoding="utf-8") as f:
            json.dump(self.problem, f)

    @staticmethod
    def _random_letters_upper(n):
        letters = ""
        all_letters = string.ascii_uppercase
        while len(letters) < n:
            random_letter = random.choice(all_letters)
            if random_letter not in letters:
                letters = letters + random_letter
        return letters

    @staticmethod
    def _random_letters_lower(n):
        letters = ""
        all_letters = string.ascii_lowercase
        while len(letters) < n:
            random_letter = random.choice(all_letters)
            if random_letter not in letters:
                letters = letters + random_letter
        return letters


def main():
    generator = Generator(init_count=0)
    generator.generate_problem_using_template_1(problem_number=10)
    generator.generate_problem_using_template_2(problem_number=10)
    generator.save()
    print("{} problems generated.".format(len(generator.problem)))


if __name__ == '__main__':
    main()
