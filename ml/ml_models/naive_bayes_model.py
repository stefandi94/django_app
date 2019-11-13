from sklearn.naive_bayes import GaussianNB

from . import MLModel


class NaiveBayes(MLModel):
    
    def __init__(self, **kwargs):
        super(MLModel).__init__(**kwargs)
        self.model = GaussianNB()
        self.model_name = 'NaiveBayes'
        