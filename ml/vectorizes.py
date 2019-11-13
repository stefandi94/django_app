from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Transformers:
    
    def __init__(self):
        self.transformer = None
    
    def fit(self, X):
        self.transformer.fit(X)
    
    def transform(self, X):
        return self.transformer.transform(X)
    
    def fit_transform(self, X):
        self.transformer.fit(X)
        return self.transform(X)


class TFIDF(Transformers):
    
    def __init__(self, **kwargs):
        super(Transformers).__init__()
        self.transformer = TfidfVectorizer()


class CV(Transformers):
    def __init__(self, **kwargs):
        super(Transformers).__init__()
        self.transformer = CountVectorizer()