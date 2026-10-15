from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" 

COLS = ["unit", "cycle", "op1", "op2","op3","s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"]
DROP_COLS = ['op3', 's1', 's5', 's10', 's16', 's18', 's19', 'op1', 'op2', 's6']
UP = ["s2", "s3", "s4", "s8", "s11", "s13", "s15", "s17"]
DOWN= ["s7", "s12", "s20", "s21"]
AMBIGUOUS=["s9", "s14"]
FEATURES = UP + DOWN + AMBIGUOUS

def load_fd001():
    train = pd.read_csv(DATA_DIR / "train_FD001.txt", sep = r"\s+", header = None, names = COLS)
    test = pd.read_csv(DATA_DIR / "test_FD001.txt", sep = r"\s+", header = None, names = COLS)
    rul = pd.read_csv(DATA_DIR/ "RUL_FD001.txt", sep = r"\s+", header = None, names = ["RUL"])
    rul.index = rul.index + 1
    return train, test, rul


def add_rul(df, cap=125):

    """Nur für Train-Set gedacht."""

    df = df.copy()
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df