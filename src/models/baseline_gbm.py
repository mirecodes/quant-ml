# src/models/baseline_gbm.py
import lightgbm as lgb
import pandas as pd
import numpy as np

class GBMBaseline:
    
    def __init__(self, target: str):
        self.target = target
        self.model = None
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.model = lgb.LGBMRegressor(
            n_estimators=100, # 속도를 위해 100으로 최적화
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        return self
    
    def predict(self, X):
        assert self.model is not None, "Fit model first"
        return self.model.predict(X)
