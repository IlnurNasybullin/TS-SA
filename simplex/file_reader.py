import numpy as np
import json
from simplex import FunctionType, Inequality


class FileReader:
    def __init__(self):
        self.function_types = {f_type.value: f_type for f_type in FunctionType}
        self.inequalities = {inequality_type.value: inequality_type for inequality_type in Inequality}

    def _get_f_type(self, data):
        if 'f_type' not in data:
            return FunctionType.MIN

        f_type = data['f_type']

        if f_type is None:
            return FunctionType.MIN
        return self.function_types[f_type]

    def _get_inequalities(self, data):
        if 'inequalities' not in data:
            return None
        inequalities = data['inequalities']

        if inequalities is None:
            return None

        ineq = []
        for inequality in inequalities:
            ineq.append(self.inequalities[inequality])

        return ineq

    @staticmethod
    def _get_normalized_x(data):
        if 'normalized_x' not in data:
            return None
        normalized_x = data['normalized_x']

        if normalized_x is None:
            return None

        norm_x = []

        for n in normalized_x:
            str_boolean = n.lower()
            if str_boolean == 'true':
                norm_x.append(True)
                continue

            if str_boolean == 'false':
                norm_x.append(False)
                continue

            raise ValueError("'It's unknown boolean expression ", str_boolean)
        return norm_x

    def read_data(self, filename):
        with open(filename, 'r') as fp:
            data = json.load(fp)

        data['A'] = np.array(data['A'], dtype=float)
        data['B'] = np.array(data['B'], dtype=float)
        data['C'] = np.array(data['C'], dtype=float)
        data['f_type'] = self._get_f_type(data)
        data['inequalities'] = self._get_inequalities(data)
        data['normalized_x'] = self._get_normalized_x(data)

        return data
