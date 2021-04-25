from file_reader import FileReader
from simplex import Simplex
from analysis import Analysis
import pandas as pd
import matplotlib.pyplot as plt

def check_range_iterations(analysis, C, i, c_i, iter_counts, key):
    C[i] = c_i
    print("C = ", C)
    (f_x, X), meta_data = analysis.replace_C(C)
    print("X =",  X)
    print("f(X) = ", f_x)
    iter_counts[key].append(meta_data.meta_dict["iter_count"])

def show_hist(hist_data, indexes):
    df = pd.DataFrame(hist_data, index=indexes)
    df.plot(kind='bar')
    plt.savefig('hist.png', format='png')
    plt.show()

def analysis(simplex, C):
    analysis = Analysis(simplex)
    iter_counts = {
        "before left range": [],
        "left range": [],
        "middle": [],
        "right range": [],
        "after right range": []
    }
    for i in range(len(C)):
        c_i = C[i]
        left_border, right_border = analysis.c_i_range(i + 1)
        print("Optimal range for c_" + str(i + 1) + " : " + str(left_border) + " <= c_" + str(i + 1) + " <= " + str(right_border))
        check_range_iterations(analysis, C, i, left_border - 1, iter_counts, "before left range")
        check_range_iterations(analysis, C, i, left_border, iter_counts, "left range")
        check_range_iterations(analysis, C, i, (left_border + right_border) / 2, iter_counts, "middle")
        check_range_iterations(analysis, C, i, right_border, iter_counts, "right range")
        check_range_iterations(analysis, C, i, right_border + 1, iter_counts, "after right range")
        C[i] = c_i

    show_hist(iter_counts, indexes=["c_" + str(i + 1) for i in range(len(C))])

if __name__ == "__main__":
    simplex_data = FileReader().read_data("data.json")
    A = simplex_data['A']
    B = simplex_data['B']
    C = simplex_data['C']
    inequalities = simplex_data['inequalities']
    f_type = simplex_data['f_type']
    normalized_x = simplex_data['normalized_x']

    simplex = Simplex()
    f_x, X = simplex.solve(A, B, C, inequalities=inequalities, f_type=f_type, normalized_x=normalized_x, log=False)
    print("C = ", C)
    print("X = ", X)
    print("f(X) = ", f_x)
    analysis(simplex, C)
