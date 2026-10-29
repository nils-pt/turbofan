from src.data import FEATURES
import pandas as pd
import numpy as np


def steigung(werte):
    """Bekommt ein Fenster von Sensorwerten (Array), gibt die Steigung der Regressionsgeraden pro Zyklus zurück"""
    t = np.arange(len(werte))
    t_abw = t - np.mean(t)
    x_abw = werte - np.mean(werte)
    return ((t_abw*x_abw).sum()/np.square(t_abw).sum())

def add_rolling(df, cols, fenster = (5,25)):
    """Rolling-Mittelwert und -Standardabweichung je Triebwerk für jedes Fenster in `fenster`.
    
    Pro Spalte in `cols` und Fenster w entstehen `{col}_mean{w}` und `{col}_std{w}`.
    Std ist im ersten Zyklus eines Triebwerks NaN (nur ein Wert) und wird mit 0 gefüllt.
    Setzt voraus: df ist je Triebwerk (`unit`) nach `cycle` sortiert.
    Rückgabe: (Kopie von df mit den neuen Spalten, Liste der neuen Spaltennamen)"""
    df = df.copy()
    rolling_spalten =[]
    for col in cols:
        for w in fenster:
            name = f"{col}_mean{w}"
            df[name]=df.groupby("unit")[col].transform(lambda s: s.rolling(w, min_periods= 1).mean())
            rolling_spalten.append(name)

            name = f"{col}_std{w}"
            df[name]= (df.groupby("unit")[col]
            .transform(lambda s: s.rolling(w, min_periods=1).std())
            .fillna(0))
            rolling_spalten.append(name)
            
    return df, rolling_spalten

def add_diff(df, cols):
    """Trend-Feature je Spalte: kurzes minus langes Rolling-Mittel (`{col}_mean5 - {col}_mean25`).
    
    Pro Spalte in `cols` entsteht `{col}_diff5_25`.
    Setzt voraus: `add_rolling` wurde vorher mit den Fenstern 5 und 25 aufgerufen.
    Rückgabe: (Kopie von df mit den neuen Spalten, Liste der neuen Spaltennamen)"""
    df = df.copy()
    diff_spalten = []
    for col in cols:
        name = f"{col}_diff5_25"
        df[name] = df[f"{col}_mean5"]- df[f"{col}_mean25"]
        diff_spalten.append(name)
    return df, diff_spalten

def add_slope(df, cols, fenster = 25):
    """Steigung der Regressionsgeraden über die letzten `fenster` Zyklen je Triebwerk.

    Pro Spalte in `cols` entsteht `{col}_slope{fenster}`.
    Die ersten `fenster - 1` Zyklen eines Triebwerks haben noch kein volles Fenster und werden mit 0 gefüllt.
    Setzt voraus: df ist je Triebwerk nach `cycle` sortiert.
    Rückgabe: (Kopie von df mit den neuen Spalten, Liste der neuen Spaltennamen)"""

    df= df.copy()
    slope_spalten=[]
    for col in cols:
        name = f"{col}_slope{fenster}"
        df[name] = (df.groupby("unit")[col]
                    .transform(lambda s: s.rolling(fenster, min_periods=fenster).apply(steigung, raw = True))
                    .fillna(0))
        slope_spalten.append(name)

    return df, slope_spalten

def build_features(df, cols=FEATURES):
    """Baut alle Features der Basis c (98 Spalten bei den 14 Sensoren aus FEATURES).

    Reihenfolge: Rohsensoren, Rolling (mean/std 5 und 25), Differenz, Steigung – wie feature_cols_c in 04.
    Setzt voraus: df ist je Triebwerk nach `cycle` sortiert.
    Rückgabe: (Kopie von df mit allen Features, Liste der Feature-Spalten)"""

    df, rolling_namen = add_rolling(df, cols)

    df, diff_namen = add_diff(df, cols)

    df, slope_namen = add_slope(df, cols)
    
    feature_cols = cols + rolling_namen + diff_namen + slope_namen

    return df, feature_cols
