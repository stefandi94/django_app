from sklearn.svm import LinearSVC

from . import MLModel


class SVM(MLModel):
    
    def __init__(self, **kwargs):
        super(MLModel).__init__(**kwargs)
        self.model = LinearSVC()
        self.model_name = 'SVM'