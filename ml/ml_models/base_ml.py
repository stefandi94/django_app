import os
from joblib import dump, load


class MLModel:
    
    def __init__(self, **kwargs):
        self.model = None
        self.model_name = None
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        dump(self.model, os.path.join(path, f'{self.model_name}.joblib'))
    
    def load(self, path):
        self.model = load(path)
    
    def __str__(self):
        return self.model_name