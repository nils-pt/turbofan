import numpy as np

def nasa_score(y_true, y_pred, aggregation = "sum"):
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    d = y_pred-y_true
    strafe = np.where(d >= 0, np.exp(d/10)-1, np.exp(-d/13)-1)
    
    if aggregation == "sum":
        return strafe.sum()
    if aggregation == "mean":
        return strafe.mean()
    
    raise ValueError(f"Aggregation muss 'sum' oder 'mean' sein, nicht '{aggregation}")