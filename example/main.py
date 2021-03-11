import numpy as np
import json
import time
from scipy.optimize import linprog

if __name__ == "__main__":
    with open('data.json', 'r') as fp:
        data = json.load(fp)

    A_ub = data.get("A_ub", None)
    b_ub = data.get("b_ub", None)
    A_eq = data.get("A_eq", None)
    b_eq = data.get("b_eq", None)
    func = data["func"]
    c = data["c"]
    x0_bounds = (0, float("inf"))
    x1_bounds = (0, 1)
    if func == "max":
        c = np.dot(c, -1)

    start = time.time()
    res_lin = linprog(c,
                      A_eq=A_eq,
                      b_eq=b_eq,
                      A_ub=A_ub,
                      b_ub=b_ub,
                      method="simplex",
                      bounds=[x0_bounds, x1_bounds])
    if func == "max":
        res_lin.fun = np.dot(res_lin.fun, -1)

    stop = time.time()
    print("Program execution time: ", stop - start)
    print(res_lin)
