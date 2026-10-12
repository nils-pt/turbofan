from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" 

cols = ["unit", "cycle", "op1", "op2","op3","s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"]
drop_cols = ['op3', 's1', 's5', 's10', 's16', 's18', 's19', 'op1', 'op2', 's6']

def load_fd001():
    train = pd.read_csv(DATA_DIR / "train_FD001.txt", sep = r"\s+", header = None, names = cols)
    test = pd.read_csv(DATA_DIR / "test_FD001.txt", sep = r"\s+", header = None, names = cols)
    rul = pd.read_csv(DATA_DIR/ "RUL_FD001.txt", sep = r"\s+", header = None, names = ["RUL"])
    rul.index = rul.index + 1
    return train, test, rul


def add_rul(df, cap=125):
    df = df.copy()
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df