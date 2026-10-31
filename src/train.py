import numpy as np
import pandas as pd

from pathlib import Path
ERGEBNIS_DIR = Path(__file__).resolve().parents[1]/"reports"/"ergebnisse"

from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error

from src.data import load_fd001, add_rul
from src.features import build_features
from src.metrics import nasa_score

def train_model():

    train, test, rul = load_fd001()
    train = add_rul(train)

    train_feat, train_cols = build_features(train)
    test_feat, test_cols = build_features(test)

    X = train_feat[train_cols]
    y = train_feat["RUL"]

    model = LGBMRegressor(verbose = -1, n_jobs = 1, random_state =42)
    model.fit(X, y)

    last_cycle_test = test_feat.groupby("unit")[test_cols].tail(1)

    assert len(last_cycle_test) == len(rul), "Größe des Testsets stimmt nicht"
    assert test_cols == train_cols, "Spalten stimmen nicht überein."
    assert list(test_feat.groupby("unit")["unit"].tail(1)) == list(rul.index), "Reihenfolge der Triebwerke stimmen nicht überein."

    y_test = rul["RUL"].to_numpy()
    y_test_capped = np.minimum(y_test, 125)

    variants = (["uncapped", y_test], ["capped", y_test_capped])
    ergebnisse =[]

    pred = model.predict(last_cycle_test)

    for name, y_wahr in variants:

        fehler = pred - y_wahr

        ergebnisse.append({
            "variante": name,
            "rmse": root_mean_squared_error(y_wahr, pred),
            "nasa_mean": nasa_score(y_wahr, pred, aggregation="mean"),
            "nasa_sum": nasa_score(y_wahr, pred, aggregation="sum"),
            "bias": fehler.mean(),
        })

    model_scores = pd.DataFrame(ergebnisse)


    ERGEBNIS_DIR.mkdir(parents=True, exist_ok=True)
    model_scores.to_csv(ERGEBNIS_DIR / "lgbm_default_scores.csv", index = False)

    ref = pd.read_csv(ERGEBNIS_DIR/"vergleich_aus_06.csv")

    assert list(ref["variante"]) == list(model_scores["variante"]), "Varianten haben nicht die selbe Reihenfolge."

    spalten = ["rmse", "nasa_mean", "nasa_sum", "bias"]

    ref = ref[spalten]
    score_vgl = model_scores[spalten]

    assert np.allclose(score_vgl, ref), "Die Werte der beiden Läufe passen nicht."

    return model_scores

if __name__ == "__main__":
    ergebnis = train_model()
    print(ergebnis)    
