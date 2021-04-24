import numpy as np
from enum import Enum
from typing import List, Optional

import pandas as pd

"""
Enum class for constraint of function type variables
"""


class FunctionType(Enum):
    MIN = "min"
    MAX = "max"


"""
Enum class for constraint of inequalities type variables
"""


class Inequality(Enum):
    EQ = '='
    LQ = '<='
    LE = '<'
    GE = '>='
    GR = '>'


class SimplexSolveException(Exception):
    pass


class IncompatibleSimplexSolveException(SimplexSolveException):
    pass


class DifficultSimplexSolveException(SimplexSolveException):
    pass


"""
Class that implemented simplex algorithm
"""


class Simplex:

    _f_type: FunctionType
    A: np
    B: np
    C: np
    inequalities: list[Inequality]
    normalized_x: Optional[list[bool]]

    """
        Set default variables; replacing index for X-vector's constraints normalization (X-vector's constraints are
        normalized, if every x variable >= 0 (

        .. math::
            x_1 >= 0, x_2 >=0, ..., x_n >= 0
         ))
    """

    def __init__(self):
        self._a_in = {Inequality.EQ: 0.0, Inequality.LQ: 1.0, Inequality.LE: 1.0, Inequality.GE: -1.0,
                      Inequality.GR: -1.0}
        self._inv_in = {Inequality.EQ: Inequality.EQ, Inequality.LE: Inequality.GR, Inequality.LQ: Inequality.GE,
                        Inequality.GR: Inequality.LE, Inequality.GE: Inequality.LQ}

    """
        Default (canonize) inequalities for simplex algorithm (all basic constraints are equations)
    """

    @staticmethod
    def default_inequalities(count: int) -> List[Inequality]:
        return [Inequality.EQ for _ in range(count)]

    """
        Default (canonize) X-vector's constraints for (every x variable is normalized (>= 0))
    """

    @staticmethod
    def default_normalized_x(count: int) -> List[bool]:
        return [True for _ in range(count)]

    def _canonize_B(self):
        for i in range(self.B.size):
            if self.B[i] < 0:
                self.A[i] = -self.A[i]
                self.B[i] = -self.B[i]
                self.inequalities[i] = self._inv_in[self.inequalities[i]]

    def _canonize_objective_function(self):
        if self._f_type == FunctionType.MAX:
            self.C = -self.C

    def _replacing_x(self):
        c_n = 0.0
        A_n = np.zeros(shape=(self.A.shape[0], 1), dtype=float)
        for i in range(len(self.normalized_x)):
            if not self.normalized_x[i]:
                self._replacing_index = self.C.size + 1

                c_n -= self.C[i]

                A_n -= np.resize(self.A[:, i], new_shape=A_n.shape)

        if self._replacing_index != -1:
            self.C = np.hstack((self.C, np.array([c_n], dtype=float)))
            self.A = np.hstack((self.A, A_n))

    def _equalization(self):
        self._x_size = self.C.size

        A_dop = []
        for i in range(len(self.inequalities)):
            A_dop.append(self._a_in[self.inequalities[i]])

        if A_dop:
            self.C = np.hstack((self.C, np.zeros(shape=len(A_dop), dtype=float)))
            self.A = np.hstack((self.A, np.diag(A_dop)))

    def _artificial_basis(self):
        size = self.B.size
        self.A = np.hstack((self.A, np.eye(size)))
        self.c_i0_index = self.C.size

        self.C_i = np.hstack((np.zeros(self.c_i0_index, dtype=float), np.ones(size, dtype=float)))

    def _log_data_preparation(self):
        if self.log:
            xSize = self.C_i.size
            self.rows_data = np.array(["P_" + str(i) for i in range(xSize)])
            self.columns_data = np.array(["B^-1 P_" + str(i) for i in range(xSize)])

    def _canonize(self):
        self._replacing_x()
        self._canonize_objective_function()
        self._canonize_B()
        self._equalization()

        B0 = np.resize(self.B, new_shape=(self.B.size, 1))
        self.A = np.hstack((B0, self.A))
        self.C = np.hstack((np.array([0.0], dtype=float), self.C))

        self._artificial_basis()

    @staticmethod
    def _first_positive(score_Jordan_Gauss):
        for i in range(1, score_Jordan_Gauss.size):
            val = score_Jordan_Gauss[i]
            if not np.isclose(val, 0) and val > 0:
                return i

        return -1

    @staticmethod
    def _theta(A, JG_index):
        size = A.shape[0] - 1

        P0 = A[:size, 0]
        P_j = A[:size, JG_index]

        min_theta = float('inf')
        min_index = -1

        for i in range(size):
            if P_j[i] > 0.0:
                theta = P0[i] / P_j[i]
                if theta < min_theta:
                    min_theta = theta
                    min_index = i

        return min_index

    def _recalculate_A(self, input_bas_i, output_bas_i):
        self.A[output_bas_i] = self.A[output_bas_i] / self.A[output_bas_i, input_bas_i]
        for i in range(self.A.shape[0]):
            if i == output_bas_i:
                continue
            k = -self.A[i, input_bas_i]
            self.A[i] += self.A[output_bas_i] * k

    def _log_A(self):
        if self.log:
            I = pd.Index(self.rows_data[self.bas_indexes].tolist() + ["delta"])
            C = pd.Index(self.columns_data)

            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            df = pd.DataFrame(data=self.A, index=I, columns=C)
            self._log(df)

    def _recalculates_A(self, border):
        while True:
            self._log_A()
            input_index = self._first_positive(self.A[-1, :border])
            if input_index == -1:
                break
            output_index = self._theta(self.A, input_index)
            if output_index == -1:
                raise IncompatibleSimplexSolveException("The system is incompatible!")
            self._recalculate_A(input_index, output_index)
            self.bas_indexes[output_index] = input_index

    def _check_bas_indexes(self):
        for index in self.bas_indexes:
            if index >= self.C.size:
                raise DifficultSimplexSolveException("The system is difficult to solve. "
                    "It is necessary to express an artificial basis as a linear combination "
                                                     "of non-artificial bases")

    @staticmethod
    def _log(msg):
        print(msg)

    def _printf_objective_function(self, C, f_type):
        function_str = "F(x) = " + str(C[0])
        x = "x" if self._replacing_index == -1 else "y"

        size = len(C)
        for i in range(1, size):
            function_str += " + " + str(C[i]) + x + "_" + str(i)

        return function_str + " -> " + f_type.value

    def _log_calc_i(self):
        if self.log:
            self._log("artificial objective function: " + self._printf_objective_function(self.C_i, FunctionType.MIN))

    def _log_calc(self):
        if self.log:
            self._log("Objective function: " + self._printf_objective_function(self.C[:self.c_i0_index], self._f_type))

    def _calc_i(self):
        self._log_calc_i()

        self.bas_indexes = [i for i in range(self.c_i0_index, self.C_i.size)]

        C_B = self.C_i[self.bas_indexes]
        score_Jordan_Gauss = C_B.dot(self.A) - self.C_i

        self.A = np.vstack((self.A, np.resize(score_Jordan_Gauss, (1, score_Jordan_Gauss.size))))
        self._recalculates_A(self.C_i.size)

        if not np.isclose(self.A[-1, 0], [0.0]):
            raise IncompatibleSimplexSolveException("The system is incompatible!")
        self._check_bas_indexes()

    def _calc(self):
        if self.log:
            self._log("Simplex calculation with two-phase artificial basis method")

        self._calc_i()
        self.C = np.hstack((self.C, np.zeros(shape=self.B.size, dtype=float)))

        self._log_calc()

        size = len(self.bas_indexes)
        self.A[size] = self.C[self.bas_indexes].dot(self.A[:size]) - self.C

        self._recalculates_A(self.c_i0_index)

    def _solve(self, A, B, C, inequalities, f_type, normalized_x, log):
        self._replacing_index = -1
        self._f_type = f_type
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.inequalities = self.default_inequalities(self.B.size) if inequalities is None else inequalities.copy()
        self.normalized_x = self.default_normalized_x(self.C.size) if normalized_x is None else normalized_x
        self.log = log

        self._canonize()
        self._log_data_preparation()
        self._calc()

    def _get_answer(self):
        f_x = self.A[-1, 0]

        X_prep = [0 for _ in range(1, self.C.size)]
        for i in range(len(self.bas_indexes)):
            X_prep[self.bas_indexes[i]] = self.A[i][0]

        if self._f_type == FunctionType.MAX:
            f_x = -f_x
        X = X_prep[1:self._x_size + 1]
        if self._replacing_index != -1:
            for i in range(1, self._replacing_index):
                if not self.normalized_x[i - 1]:
                    X_prep[i] -= X_prep[self._replacing_index]
            X = X_prep[1:self._replacing_index]

        return f_x, X

    def solve(self, A: np, B: np, C: np, inequalities: List[Inequality] = None,
              f_type: FunctionType = FunctionType.MIN, normalized_x: List[bool] = None, log=False) \
            -> [float, List[float]]:
        self._solve(A, B, C, inequalities, f_type, normalized_x, log)

        return self._get_answer()
