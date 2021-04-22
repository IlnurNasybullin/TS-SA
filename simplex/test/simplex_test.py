import unittest
import numpy as np
import simplex.simplex as smp

from ddt import ddt, data, unpack


@ddt
class SimplexTestCase(unittest.TestCase):

    def setUp(self):
        self.simplex = smp.Simplex()

    @data(
        {'A': np.array([[-1, 1, 1, 2, -3],
                        [1, 1, 4, 1, -8],
                        [0, 1, 1, 0, -4]], dtype=float),
         'B': np.array([4, 3, -4], dtype=float),
         'C': np.array([-1, -1, 1, 3, 7], dtype=float),
         'f_x': 20.0,
         'X': [5.0, 0.0, 0.0, 6.0, 1.0]}
    )
    @unpack
    def test_solve_1(self, A, B, C, f_x, X):
        """
        .. image:: test_solve_1.png

        """

        f_x_actual, X_actual = self.simplex.solve(A, B, C)

        self.assertAlmostEqual(f_x_actual, f_x)
        np.testing.assert_almost_equal(X_actual, X)

    @data({
        'A': np.array([[1, 0],
                       [0, 1],
                       [1, 1],
                       [1, 2]], dtype=float),
        'B': np.array([40, 30, 60, 80], dtype=float),
        'C': np.array([2, 3], dtype=float),
        'inequalities': [smp.Inequality.LQ, smp.Inequality.LQ, smp.Inequality.LQ, smp.Inequality.LQ],
        'f_x': 140,
        'X': [40.0, 20.0],
        'f_type': smp.FunctionType.MAX
    })
    @unpack
    def test_solve_2(self, A, B, C, inequalities, f_x, X, f_type):

        """
        .. image:: test_solve_2.png

        """

        f_x_actual, X_actual = self.simplex.solve(A, B, C, inequalities, f_type=f_type)

        self.assertAlmostEqual(f_x_actual, f_x)
        np.testing.assert_almost_equal(X_actual, X)

    @data({
        'A': np.array([[2, -1, 1],
                       [4, -2, 1],
                       [3, 0, 1]], dtype=float),
        'B': np.array([1, -2, 5], dtype=float),
        'C': np.array([1, -1, -3], dtype=float),
        'inequalities': [smp.Inequality.LQ, smp.Inequality.GE, smp.Inequality.LQ],
        'f_x': -46/3,
        'X': [1/3, 11/3, 4.0]
    })
    @unpack
    def test_solve_3(self, A, B, C, inequalities, f_x, X):
        """
        .. image:: test_solve_3.png

        """

        f_x_actual, X_actual = self.simplex.solve(A, B, C, inequalities)

        self.assertAlmostEqual(f_x_actual, f_x)
        np.testing.assert_almost_equal(X_actual, X)

    @data({
        'A': np.array([[-2, 3],
                       [1, 1],
                       [3, -5]], dtype=float),
        'B': np.array([12, 9, 3], dtype=float),
        'C': np.array([1, -1], dtype=float),
        'inequalities': [smp.Inequality.LQ, smp.Inequality.LQ, smp.Inequality.LQ],
        'f_x': 3,
        'X': [6, 3],
        'f_type': smp.FunctionType.MAX
    })
    @unpack
    def test_solve_4(self, A, B, C, inequalities, f_x, X, f_type):
        f_x_actual, X_actual = self.simplex.solve(A, B, C, inequalities, f_type=f_type)

        self.assertAlmostEqual(f_x_actual, f_x)
        np.testing.assert_almost_equal(X_actual, X)

    @data({
        'A': np.array([[-2, 3],
                       [3, -5]]),
        'B': np.array([12, 3]),
        'C': np.array([1, 1]),
        'inequalities': [smp.Inequality.LQ, smp.Inequality.LQ],
        'f_type': smp.FunctionType.MAX,
        'exception': smp.IncompatibleSimplexSolveException
    })
    @unpack
    def test_solve_5(self, A, B, C, inequalities, f_type, exception):
        self.assertRaises(exception, self.simplex.solve, A, B, C, inequalities, f_type)

    @data({
        'A': np.array([[-2, 3],
                       [1, 1],
                       [3, -5]], dtype=float),
        'B': np.array([12, 9, 3], dtype=float),
        'C': np.array([1, -1], dtype=float),
        'inequalities': [smp.Inequality.LQ, smp.Inequality.LQ, smp.Inequality.GE],
        'f_type': smp.FunctionType.MAX,
        'f_x': 9,
        'X': [9, 0]
    })
    @unpack
    def test_solve_6(self, A, B, C, inequalities, f_type, f_x, X):
        f_x_actual, X_actual = self.simplex.solve(A, B, C, inequalities, f_type=f_type)

        self.assertAlmostEqual(f_x_actual, f_x)
        np.testing.assert_almost_equal(X_actual, X)

    @data({
        'A': np.array([[-2, 3],
                       [1, 1],
                       [1, -1]], dtype=float),
        'B': np.array([12, 9, 3], dtype=float),
        'C': np.array([1, -1], dtype=float),
        'inequalities': [smp.Inequality.LQ, smp.Inequality.LQ, smp.Inequality.LQ],
        'f_type': smp.FunctionType.MAX,
        'f_x': 3
    })
    @unpack
    def test_solve_7(self, A, B, C, inequalities, f_type, f_x):
        f_x_actual, X_actual = self.simplex.solve(A, B, C, inequalities, f_type=f_type)
        self.assertAlmostEqual(f_x_actual, f_x)

    @data({
        'A': np.array([[2, -1, 1],
                       [4, -2, 1],
                       [3, 0, 1]], dtype=float),
        'B': np.array([1, -2, 5]),
        'C': np.array([1, -1, -3]),
        'inequalities': [smp.Inequality.LQ, smp.Inequality.GE, smp.Inequality.LQ],
        'f_x': -46./3. ,
        'X': [1./3., 11./3., 4]
    })
    @unpack
    def test_solve_8(self, A, B, C, inequalities, f_x, X):
        f_x_actual, X_actual = self.simplex.solve(A, B, C, inequalities)

        self.assertAlmostEqual(f_x_actual, f_x)
        np.testing.assert_almost_equal(X_actual, X)

    @data({
        'A': np.array([[1, 3, 2, 2],
                       [2, 2, 1, 1]], dtype=float),
        'B': np.array([3, 3], dtype=float),
        'C': np.array([5, 3, 4, -1], dtype=float),
        'f_type': smp.FunctionType.MAX,
        'f_x': 9,
        'X': [1, 0, 1, 0]
    })
    @unpack
    def test_solve_9(self, A, B, C, f_type, f_x, X):
        f_x_actual, X_actual = self.simplex.solve(A, B, C, f_type=f_type)

        self.assertAlmostEqual(f_x_actual, f_x)
        np.testing.assert_almost_equal(X_actual, X)

    @data({
        'A': np.array([[1, 1],
                       [0, -1]], dtype=float),
        'B': np.array([1, 3], dtype=float),
        'C': np.array([-1, 2], dtype=float),
        'inequalities': [smp.Inequality.LQ, smp.Inequality.LQ],
        'normalized_x': [False, False],
        'f_x': -10,
        'X': [4, -3]
    })
    @unpack
    def test_solve_10(self, A, B, C, inequalities, normalized_x, f_x, X):
        f_x_actual, X_actual = self.simplex.solve(A, B, C, inequalities, normalized_x=normalized_x, log=True)

        self.assertAlmostEqual(f_x_actual, f_x)
        np.testing.assert_almost_equal(X_actual, X)


if __name__ == '__main__':
    unittest.main()
