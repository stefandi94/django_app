from ml.vectorizes import CV, TFIDF
from .ml_models import NaiveBayes, RandomForest, SVM, XGB


def get_ml_model(model_name):
    model_name = model_name.lower()
    
    if model_name == 'naivebayes':
        return NaiveBayes()
    
    elif model_name == 'randomforest':
        return RandomForest()
    
    elif model_name == 'svm':
        return SVM()
    
    elif model_name == 'xgb':
        return XGB()
    
    else:
        raise ValueError(f'Unknown model name {model_name}. '
                         f'Please choose from: naivebayes, randomforest, svm or xgb')


def get_transformer(transformer_name):
    transformer_name = transformer_name.lower()

    if transformer_name == 'cv':
        transformer = CV()

    elif transformer_name == 'tfidf':
        transformer = TFIDF()
    
    else:
        raise ValueError(f'Unknown transformer name {transformer_name}. '
                         f'Please choose from: cv and tfidf')
    
    return transformer