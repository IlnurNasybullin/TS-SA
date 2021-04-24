from simplex import Simplex, FunctionType, Inequality
import numpy as np


class Analysis:
    def __init__(self, simplex):
        self.simplex = simplex

    def _is_normalized_x(self, i):
        return self.simplex.normalized_x[i - 1]

    def _is_basis_P_i(self, i):
        return i in self.bas_indexes

    def _basis_c_delta(self, i):
        left_range = -float('Inf')
        right_range = float('Inf')
        basis_index = self.simplex.bas_indexes.index(i)
        for k in range(1, self.simplex.c_i0_index):
            if k == i:
                continue
            Jordan_Gauss_score = self.simplex.A[-1, k]
            a_ik = self.simplex.A[basis_index, k]
            if a_ik == 0:
                continue

            val = -Jordan_Gauss_score / a_ik
            # if self.simplex._f_type == FunctionType.MIN:
            #     val = -val

            is_positive = a_ik > 0
            is_min = self.simplex._f_type == FunctionType.MAX

            if is_positive ^ is_min:
                right_range = min(right_range, val)
            else:
                left_range = max(left_range, val)
        return left_range, right_range

    def _not_basis_c_delta(self, i):
        Jordan_Gauss_score = self.simplex.A[-1, i]
        val = Jordan_Gauss_score
        if self.simplex._f_type == FunctionType.MAX:
            val = -val
        return val, float('Inf')

    def _norm_c_delta(self, i):
        if self._is_basis_P_i(i):
            return self._basis_c_delta(i)
        else:
            return self._not_basis_c_delta(i)

    def _not_basis_not_basis_c_delta(self, i, j):
        l_i, r_i = self._not_basis_c_delta(i)
        l_j, r_j = self._not_basis_c_delta(j)
        return max(l_i, l_j), min(r_i, r_j)

    def _basis_not_basis_c_delta(self, i, j):
        left_range = -float('Inf')
        right_range = float('Inf')
        basis_index = self.simplex.bas_indexes.index(i)
        for k in range(1, self.simplex.c_i0_index):
            if k == i:
                continue
            Jordan_Gauss_score = self.simplex.A[-1, k]
            a_ik = self.simplex.A[basis_index, k]
            if k == j:
                a_ik -= 1

            if a_ik == 0:
                continue

            val = Jordan_Gauss_score / a_ik
            is_positive = a_ik > 0
            is_min = self.simplex._f_type == FunctionType.MAX

            if is_positive ^ is_min:
                right_range = min(right_range, val)
            else:
                left_range = max(left_range, val)
        return left_range, right_range

    def _basis_and_basis_c_delta(self, i, j):
        l_1, r_1 = self._basis_not_basis_c_delta(i, j)
        l_2, r_2 = self._basis_not_basis_c_delta(j, i)
        return max(l_1, l_2), min(r_1, r_2)

    def _not_norm_c_delta(self, i):
        j = self.simplex._replacing_index
        if self._is_basis_P_i(i):
            if self._is_basis_P_i(j):
                return self._basis_and_basis_c_delta(i, j)
            else:
                return self._basis_not_basis_c_delta(i, j)
        else:
            if self._is_basis_P_i(j):
                return self._basis_not_basis_c_delta(j, i)
            else:
                return self._not_basis_not_basis_c_delta(i, j)

    def c_delta(self, i):
        self.bas_indexes = set(self.simplex.bas_indexes)
        if self._is_normalized_x(i):
            return self._norm_c_delta(i)
        else:
            return self._not_norm_c_delta(i)

    def c_i_range(self, i):
        c_i = self.simplex.C[i]
        left_range, right_range = self.c_delta(i)
        return c_i + left_range, c_i + right_range


A = np.array([[1, 4],
              [3, 5],
              [6, 1],
              [7, 2]], dtype=float)
B = np.array([3, 5, 2, 10], dtype=float)
C = np.array([1000, 1500], dtype=float)
f_type = FunctionType.MIN
inequalities = [Inequality.GE, Inequality.GE, Inequality.GE, Inequality.GE]

smp = Simplex()
f_x, X = smp.solve(A, B, C, f_type=f_type, inequalities=inequalities)
anl = Analysis(smp)
print(anl.c_i_range(1))
print(anl.c_i_range(2))
