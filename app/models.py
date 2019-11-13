import os

from django.db import models

from ml.settings import MODEL_DIR, DATA_DIR

MODEL_CHOICES = (('RandomForest', 'RandomForest'),
                 ('SVM', 'SVM'),
                 ('XGB', 'XGB'),
                 ('NaiveBayes', 'NaiveBayes'))

VECTORIZER_CHOICES = (('TfIdf', 'TfIdf'),
                      ('CV', 'CV'))

SAVE_CHOICE = ((True, True),
               (False, False))

MODEL_TYPE = (('Machine Learning', 'Machine Learning'),
              ('Deep Learning', 'Deep Learning'))


class TrainModel(models.Model):
    train_data = models.fields.Field(default=os.path.join(DATA_DIR, 'train_text'))
    train_labels = models.fields.Field(default=os.path.join(DATA_DIR, 'train_labels'))

    model_type = models.fields.CharField(choices=MODEL_TYPE)

    save_model = models.fields.BooleanField(choices=SAVE_CHOICE)
    save_encoder = models.fields.BooleanField(choices=SAVE_CHOICE)
    save_transformer = models.fields.BooleanField(choices=SAVE_CHOICE)

    model_name = models.CharField(max_length=64, choices=MODEL_CHOICES)
    vectorizer = models.CharField(max_length=64, choices=VECTORIZER_CHOICES)
    save_path = models.CharField(max_length=256, default=MODEL_DIR)


class PredictModel(models.Model):
    text = models.TextField()
    model_name = models.CharField(max_length=64, choices=MODEL_CHOICES)
    vectorizer = models.CharField(max_length=64, choices=VECTORIZER_CHOICES)
    load_path = models.CharField(max_length=256, default=MODEL_DIR)

    def __str__(self):
        return self.text
