import pandas as pd

class Matcher:
    def __init__(self):
        pass
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict match vs. no-match.
        Tuned toward precision, with explicit path for predicting "no match".
        """
        pass

def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate matcher output into one row per Source 1 entity format.
    """
    pass
