from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from ml.get_ml_model import get_ml_model, get_transformer
from ml.load_data import load_data, pickle_load, pickle_save
from ml.preprocessing import remove_special_characters
from ml.settings import MODEL_DIR

vectorizer = 'TfIdf'
model_name = 'RandomForest'
le = 'label_encoder'
load_model = False
save_model = True

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    
    X_train = [remove_special_characters(x) for x in X_train]
    X_test = [remove_special_characters(x) for x in X_test]
    
    if load_model:
        label_encoder = pickle_load(MODEL_DIR, f'{le}')
        model = pickle_load(MODEL_DIR, f'{model_name}')
        transformer = pickle_load(MODEL_DIR, f'{vectorizer}')
    else:
        label_encoder = LabelEncoder()
        model = get_ml_model(model_name)
        y_train = label_encoder.fit_transform(y_train)
        transformer = get_transformer(vectorizer)
        transformer.fit(X_train)
        word_count_vector = transformer.transform(X_train)
        model.fit(word_count_vector, y_train)
        
    test_count_vector = transformer.transform(X_test)
    y_test = label_encoder.transform(y_test)
    
    if save_model:
        pickle_save(model, MODEL_DIR, f'{model_name}')
        pickle_save(label_encoder, MODEL_DIR, f'{le}')
        pickle_save(transformer, MODEL_DIR, f'{vectorizer}')
        
    y_pred = model.predict(test_count_vector)
    print(f'Accuracy is {accuracy_score(y_test, y_pred)}')
