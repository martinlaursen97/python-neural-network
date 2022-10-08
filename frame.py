import numpy as np


class Frame:
    def __init__(self, file_path):
        self.file_path = file_path
        self.x = []
        self.y = []

    def transform(self, input_idx, output_from_idx=0):
        file = open(self.file_path, 'r')
        c = 0

        temp_y = []

        while True:
            c += 1

            line = file.readline()
            line_split = line.split(',')

            if not line:
                break

            self.x.append([int(i) for i in line_split[output_from_idx:]])
            temp_y.append(int(line_split[input_idx]))
        self.y = temp_y

    def get_max(self):
        max = 0
        for i in self.x:
            for j in i:
                if j > max:
                    max = j

        return max

    def normalize(self):
        max = self.get_max()
        self.x = [[i / max for i in j] for j in self.x]

    def get_x(self):
        return np.array(self.x)

    def get_y(self):
        return np.array(self.y)
