import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.base import clone
from src.metrics import nasa_score


def run_cv(model, X, y, splits):

    """GroupKFold-Kreuzvalidierung: pro Fold wird eine frische Kopie von `model`
    auf dem Train-Teil gefittet und auf dem Val-Teil bewertet.

    splits: Liste von (train_idx, val_idx)-Tupeln, z. B. list(gkf.split(X, y, groups)),
            kein Generator, da dieser nach einem Durchlauf leer ist.
    Rückgabe: DataFrame mit einer Zeile pro Fold (RMSE Train/Val, NASA-Score, Fehlerdiagnose).
    """
    
    ergebnisse = []

    for fold, (train_idx, val_idx) in enumerate(splits):

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        pred_tr = m.predict(X_tr)
        pred_val = m.predict(X_val)
        d = pred_val - y_val

        ergebnisse.append({

            "fold": fold,
            "rmse_train": root_mean_squared_error(y_tr, pred_tr),
            "rmse_val": root_mean_squared_error(y_val, pred_val),
            "nasa_val": nasa_score(y_val, pred_val, "mean"),
            "bias": d.mean(),
            "anteil_spaet": (d>0).mean(),
            "max_spaet": d.max(),
        })

    cv = pd.DataFrame(ergebnisse)

    return cv


