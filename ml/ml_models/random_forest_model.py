from sklearn.ensemble import RandomForestClassifier

from . import MLModel


class RandomForest(MLModel):
    
    def __init__(self, **kwargs):
        super(MLModel).__init__(**kwargs)
        self.model = RandomForestClassifier()
        self.model_name = 'RandomForest'