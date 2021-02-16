import pandas as pd
from typing import List


class RockGrader:
    def __init__(self):
        df = pd.read_csv('./data/rock_grade_label.csv', usecols=['position', 'grade'])
        self.positions = df['position'].tolist()
        self.grades = df['grade'].tolist(    )

    def grade(self, position: float):
        for i, value in enumerate(self.positions):
            if position <= value:
                return ((self.grades[i] - self.grades[i - 1]) / (self.positions[i] - self.positions[i - 1]) *
                        (position - self.positions[i])) + self.grades[i]

    def grade_to_csv(self, positions: List[float]):
        grades = []
        for position in positions:
            grades.append(self.grade(position))

        data = {'position': positions, 'grade': grades}
        df = pd.DataFrame(data)
        df.to_csv('./data/youngdeok_rock_grade.csv', index_label="id")


if __name__ == '__main__':
    df = pd.read_csv('./data/youngdeok_data.csv', usecols=['position'])
    positions = df['position'].tolist()

    rock_grader = RockGrader()
    rock_grader.grade_to_csv(positions)
