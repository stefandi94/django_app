from django.contrib import messages
from django.shortcuts import render
from rest_framework import status
# from app.models import TextModel
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from sklearn.preprocessing import LabelEncoder

from app.forms import TrainForm, PredictForm
from app.serializers import PredictSerializer, TrainSerializer
from ml.get_ml_model import get_ml_model, get_transformer
from ml.load_data import pickle_load, pickle_save, load_file


# Create your views here.
class Train(GenericAPIView):
    serializer_class = TrainSerializer
    
    @staticmethod
    def post(request):
        try:
            my_data = request.data
            train_data_path = my_data['train_data']
            train_labels_path = my_data['train_labels']
            
            X_train = load_file(train_data_path)
            y_train = load_file(train_labels_path)
            
            model_name = my_data['model_name']
            vectorizer = my_data['vectorizer']
            save_path = my_data['save_path']
            
            label_encoder = LabelEncoder()
            model = get_ml_model(model_name)
            y_train = label_encoder.fit_transform(y_train)
            transformer = get_transformer(vectorizer)
            transformer.fit(X_train)
            word_count_vector = transformer.transform(X_train)
            model.fit(word_count_vector, y_train)
            
            pickle_save(model, save_path, f'{model_name}')
            pickle_save(label_encoder, save_path, '{label_encoder}')
            pickle_save(transformer, save_path, f'{vectorizer}')
        
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
        
        return Response(f'Your model is: {model} is trained and saved at {save_path}',
                        status=status.HTTP_200_OK)


class Predict(GenericAPIView):
    serializer_class = PredictSerializer
    
    @staticmethod
    def post(request):
        try:
            my_data = request.data
            model_name = my_data['model_name']
            vectorizer = my_data['vectorizer']
            load_path = my_data['load_path']
            text = [my_data['text']]
            label_encoder = 'label_encoder'
            
            le = pickle_load(load_path, f'{label_encoder}')
            model = pickle_load(load_path, f'{model_name}')
            transformer = pickle_load(load_path, f'{vectorizer}')
            
            test_count_vector = transformer.transform(text)
            y_pred = model.predict(test_count_vector)
            prediction = le.inverse_transform(y_pred)[0].strip()
            
            return Response(f'Your comment type is: {prediction}',
                            status=status.HTTP_200_OK)
        
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
        

def predict(my_data):
    try:
        model_name = my_data['model_name']
        vectorizer = my_data['vectorizer']
        load_path = my_data['load_path']
        text = [my_data['text']]
        label_encoder = 'label_encoder'
    
        le = pickle_load(load_path, f'{label_encoder}')
        model = pickle_load(load_path, f'{model_name}')
        transformer = pickle_load(load_path, f'{vectorizer}')
    
        test_count_vector = transformer.transform(text)
        y_pred = model.predict(test_count_vector)
        prediction = le.inverse_transform(y_pred)[0].strip()
        
        return f'Your comment type is: {prediction}'
    
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def text_predict(request):
    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid():
            # text = form.cleaned_data['text']
            # model_name = form.cleaned_data['model_name']
            # vectorizer = form.cleaned_data['vectorizer']
            # load_path = form.cleaned_data['load_path']
            # text = [form.cleaned_data['text']]
            # label_encoder = 'label_encoder'
            my_dict = request.POST.dict()
            result = predict(my_dict)
            print(result)
            messages.success(request, f"Sentiment analysis: \n{result}")
    
    form = PredictForm()
    return render(request, 'predict_form.html', {'form': form})