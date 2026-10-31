import pandas as pd

from src.data import add_rul

def test_letzter_zyklus_hat_rul_0():
    df = pd.DataFrame({
        "unit": [1,1,1,2,2,2,2],
        "cycle": [1,2,3,1,2,3,4],
    })
    
    df = add_rul(df)
    
    assert list(df.groupby("unit")["RUL"].last()) == [0,0], "letzter Zyklus von mind. einem Triebwerk hat RUL ungleich 0"

def test_kappung():
    df = pd.DataFrame({
            "unit": [1,1,1,2,2,2,2],
            "cycle": [1,2,3,1,2,3,4],
        })
        
    df = add_rul(df, cap = 2)

    assert list(df["RUL"]) == [2,1,0,2,2,1,0], "RUL-Werte mit cap=2 stimmen nicht"

def test_eingabe_unveraendert():

    df = pd.DataFrame({
                "unit": [1,1,1,2,2,2,2],
                "cycle": [1,2,3,1,2,3,4],
            })

    df2 = add_rul(df)

    assert "RUL" not in df.columns, "df wurde um RUL-Spalte ergänzt"   