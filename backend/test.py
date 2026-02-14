
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\fs201\Downloads\RegVAR\output\eval_results.tsv', sep='\t', header=None)

# 将相关列转换为字符串
cols = [df[i].astype(str) for i in range(4)]
# 使用numpy的char.add进行向量化操作，性能更高
result = np.char.add(np.char.add(np.char.add(cols[0], ':'), cols[1]),
                     np.char.add(np.char.add('_', cols[2]),
                                 np.char.add('>', cols[3])))
df['final'] = result
results = []
for _, row in df.iterrows():
    print(row)
    result_item = {
        "id": (row['final']),
        "riskScore": (row[5])
    }
    results.append(result_item)
print(results)