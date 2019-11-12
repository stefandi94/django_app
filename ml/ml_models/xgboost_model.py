from xgboost import XGBClassifier

from . import MLModel


class XGB(MLModel):
    
    def __init__(self, **kwargs):
        super(MLModel).__init__(**kwargs)
        self.model = XGBClassifier()
        self.model_name = 'XGB'