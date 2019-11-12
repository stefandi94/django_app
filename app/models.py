from django.db import models

# Create your models here.


# class TextModel(models.Model):
#     text = models.TextField()
#     model_name = models.CharField(max_length=64)
#     vectorizer = models.CharField(max_length=64)
#
#     def __str__(self):
#         return self.text
from django import forms

from ml.settings import MODEL_DIR

MODEL_CHOICES = (('RandomForest', 'RandomForest'),
                      ('SVM', 'SVM'),
                      ('XGB', 'XGB'),
                      ('NaiveBayes', 'NaiveBayes'))

VECTORIZER_CHOICES = (('TfIdf', 'TfIdf'),
                      ('CV', 'CV'))


class TrainModel(models.Model):
    train_data = models.fields.FilePathField(path=MODEL_DIR)
    train_labels = models.fields.FilePathField(path=MODEL_DIR)
    model_name = models.CharField(max_length=64, choices=MODEL_CHOICES)
    vectorizer = models.CharField(max_length=64, choices=VECTORIZER_CHOICES)
    save_path = models.CharField(max_length=256, default=MODEL_DIR)


class PredictModel(models.Model):
    text = models.TextField()
    # text = forms.CharField(max_length=512,
    #                        widget=forms.TextInput(attrs={"placeholder": "Enter comment about movies"}))
    model_name = models.CharField(max_length=64, choices=MODEL_CHOICES)
    vectorizer = models.CharField(max_length=64, choices=VECTORIZER_CHOICES)
    load_path = models.CharField(max_length=256, default=MODEL_DIR)
    
    def __str__(self):
        return self.text