import pandas as pd
import numpy as np

from src.features import add_rolling

def test_no_future_data():
    df = pd.DataFrame({
        "unit": [1,1,1,1,1,1],
        "cycle": [1,2,3,4,5,6],
        "s2": [1,2,3,4,5,6],
    })

    rolling_feat1, feature_cols1 = add_rolling(df, ["s2"])

    df1 = df.copy()
    df1["s2"] = [1,2,3,4,5,100]

    rolling_feat2, feature_cols2 = add_rolling(df1, ["s2"])

    maske_1_5 = rolling_feat1["cycle"].between(1,5)

    assert np.allclose(rolling_feat1.loc[maske_1_5, "s2_mean5"], rolling_feat2.loc[maske_1_5, "s2_mean5"]), "s2_mean5 hat sich geändert, Zukunftsdaten abgebildet"
    
    maske_6 = rolling_feat1["cycle"]==6

    assert not np.allclose(rolling_feat1.loc[maske_6, "s2_mean5"], rolling_feat2.loc[maske_6, "s2_mean5"]), "s2_mean5 hat die selben Werte (keine Änderung übernommen)"

def test_fenster_triebwerksgrenze():
    df = pd.DataFrame({
        "unit": [1,1,1,1,1,1,2,2,2],
        "cycle": [1,2,3,4,5,6,1,2,3],
        "s2": [1,2,3,4,5,6,100,100,100]
    })

    feat, feat_col = add_rolling(df, ["s2"])

    maske_1 = (feat["unit"]==1) & (feat["cycle"] == 6)

    assert maske_1.sum() == 1, "Maske trifft keine Zeile"

    assert np.allclose(feat.loc[maske_1, "s2_mean5"], 4.0), "Fenster enthält nicht genau 5 Werte"

    maske_2 = (feat["cycle"] == 1) & (feat["unit"] == 2)

    assert maske_2.sum() == 1, "Maske trifft keinen Wert"

    assert np.allclose(feat.loc[maske_2, "s2_mean5"], 100), "Fenster reicht über Triebwerksgrenze"