from simplex import Simplex, FunctionType, Inequality, IncompatibleSimplexSolveException
import numpy as np


class Analysis:

    class Meta:
        def __init__(self):
            self.meta_dict = {}

    class SimplexAnalysis(Simplex):

        def __init__(self, simplex):
            self.A = simplex.A.copy()
            self.B = simplex.B.copy()
            self.C = simplex.C.copy()
            self._f_type = simplex._f_type
            self.inequalities = simplex.inequalities.copy()
            self.log = simplex.log
            self._replacing_index = simplex._replacing_index
            self.c_i0_index = simplex.c_i0_index
            self.bas_indexes = simplex.bas_indexes.copy()
            self.normalized_x = simplex.normalized_x.copy()
            if simplex.log:
                self.rows_data = simplex.rows_data
                self.columns_data = simplex.columns_data
            self.C_i = simplex.C_i.copy()
            self._x_size = simplex._x_size

        def _recalculates_A(self, border):
            iter_count = 0

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
                iter_count += 1

            self.meta_data = Analysis.Meta()
            self.meta_data.meta_dict["iter_count"] = iter_count
    
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

    def _get_simplex_copy(self):
        return Analysis.SimplexAnalysis(self.simplex)

    def _replacing_x_canonize_C(self, C):
        if self.simplex._replacing_index == -1:
            return C
        
        c_i = 0
        for i in range(len(C)):
            if not self.simplex.normalized_x[i]:
                c_i -= C[i]
        
        C = np.hstack((C, np.array([c_i], dtype=float)))
        return C

    def _function_type_C(self, C):
        if self.simplex._f_type == FunctionType.MAX:
            C = -C
        canonized_C = self.simplex.C.copy()
        for i in range(1, C.shape[0] + 1):
            canonized_C[i] = C[i - 1]

        return canonized_C

    def _get_canonized_C(self, C):
        C_ar = self._replacing_x_canonize_C(C)
        return self._function_type_C(C_ar)
        
    def _resolve_C(self, simplex):
        size = len(simplex.bas_indexes)
        simplex.A[size] = simplex.C[simplex.bas_indexes].dot(simplex.A[:size]) - simplex.C
        simplex._recalculates_A(simplex.c_i0_index)

    def replace_C(self, C):
        simplex = self._get_simplex_copy()
        simplex.C = self._get_canonized_C(C.copy())
        self._resolve_C(simplex)
        return simplex._get_answer(), simplex.meta_data


A = np.array([[-1, 1],
              [0, 1],
              [1, 0]], dtype=float)
B = np.array([2, 1, 3], dtype=float)
C = np.array([6, 10], dtype=float)
f_type = FunctionType.MAX
inequalities = [Inequality.LQ, Inequality.LQ, Inequality.LQ]

smp = Simplex()
f_x, X = smp.solve(A, B, C, f_type=f_type, inequalities=inequalities, log=True)
print(f_x)
print(X)
anl = Analysis(smp)
(f_x, X), meta_data = anl.replace_C(np.array([-1, 10], dtype=float))
print(f_x)
print(X)
print(meta_data.meta_dict)
