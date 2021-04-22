import pandas as pd
import numpy as np
import seaborn as sns


s = np.array(["C_bas"] + ["B^-1 P_" + str(i) for i in range(3)])

print(s.tolist() + ["str"])
i = [2]

I = pd.Index(s)
C = pd.Index(["col0", "col1", "col2"])
df = pd.DataFrame(data=np.random.rand(4,3), index=I, columns=C)

df = df.rename(index={"a": "23"})
print(I)
print(df)