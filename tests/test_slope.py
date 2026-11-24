import pandas as pd
import numpy as np

from src.features import add_slope

def mini_df():
    df = pd.DataFrame({
        "unit": [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2],
        "s2": [3,5,7,9,11,13,15,17,97,94,91,88,85,82,79,76],
        "cycle": [1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8]
    })
    df = df.copy()
    return df

def test_slope_linear_values():
    df = mini_df()
    df_neu, neue_spalten = add_slope(df, ["s2"], fenster=5)
    spalte = neue_spalten[0]

    for unit, erwartet in [(1, 2.0), (2, -3.0)]:
        maske_voll = (df_neu["unit"] == unit) & (df_neu["cycle"] >= 5)
        maske_leer = (df_neu["unit"] == unit) & (df_neu["cycle"] < 5)

        assert maske_voll.sum() == 4, "full-window mask does not hit exactly 4 rows"
        assert maske_leer.sum() == 4, "placeholder mask does not hit exactly 4 rows"

        werte_voll = df_neu.loc[maske_voll, spalte]
        werte_leer = df_neu.loc[maske_leer, spalte]

        assert np.allclose(werte_voll, erwartet), f"unit {unit}: slope is not {erwartet}"
        assert (werte_leer == 0).all(), f"unit {unit}: placeholder is not 0"

def test_slope_no_future_data():
    k = 7
    fenster = 5
    df_orig = mini_df()
    df_mod = mini_df()
    
    mask = (df_mod["unit"] == 1) &(df_mod["cycle"]== k)

    assert mask.sum() == 1, "Maske trifft nicht die erwartete Zeilenanzahl"

    df_mod.loc[mask, ["s2"]] = 50

    df_1, neue_spalten = add_slope(df_orig, ["s2"], fenster=fenster)
    df_2, neue_spalten = add_slope(df_mod, ["s2"], fenster=fenster)

    spalte = neue_spalten[0]

    mask_vor = (df_1["cycle"]<k) & (df_1["unit"] == 1) & (df_1["cycle"]>= fenster)

    assert mask_vor.sum() == k - fenster, "Zeilenanzahl der Maske vor cycle = 7 stimmt nicht"

    werte_vor_1 = df_1.loc[mask_vor, spalte]
    werte_vor_2 = df_2.loc[mask_vor, spalte]

    assert np.allclose(werte_vor_1,werte_vor_2), "Steigung ist bereits vor cycle = k verändert"
    
    assert not np.isclose(df_2.loc[mask, spalte], df_1.loc[mask, spalte]), "Steigung von Triebwerk 1 an Stelle k ist gleich"

