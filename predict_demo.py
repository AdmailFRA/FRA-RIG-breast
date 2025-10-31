import joblib, pandas as pd
from sklearn.datasets import load_breast_cancer

PATH = "fra_rig_breast.joblib"  # файл лежит в корне репозитория
system = joblib.load(PATH)

def rig_predict_proba(system, X_new: pd.DataFrame):
    import numpy as np
    prep = system["prep"]
    head = system["head"]
    num_cols = system["num_cols"]

    # восстановим веса
    flat_w = {}
    for b in ['R','I','G']:
        for c, v in system['block_weights'][b].items():
            flat_w[c] = float(v)

    Z = prep.named_transformers_['num'].transform(X_new[num_cols])
    col_idx = {c: i for i, c in enumerate(num_cols)}

    def block_sum(block):
        w = np.array([flat_w[c] for c in block], float)
        Zb = Z[:, [col_idx[c] for c in block]]
        return (Zb * w).sum(axis=1)

    S = pd.DataFrame({
        'S_R': block_sum(system['blocks']['R']),
        'S_I': block_sum(system['blocks']['I']),
        'S_G': block_sum(system['blocks']['G']),
    }, index=X_new.index)

    proba = head.predict_proba(S)[:, 1]
    return proba, S

# Демонстрация на sklearn-датасете (первые 3 записи)
ds = load_breast_cancer()
X = pd.DataFrame(ds.data, columns=ds.feature_names)
proba, S = rig_predict_proba(system, X.iloc[:3])
print("Probabilities:", [round(float(x), 3) for x in proba])
print(S.round(3))
