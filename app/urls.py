from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

from app import views
from app.views import Predict, Train


urlpatterns = [
    path('train_form/', views.text_train, name="train_form"),
    path('predict_form/', views.text_predict, name="predict_form"),
    path('train/', Train.as_view(), name="train"),
    path('predict/', Predict.as_view(), name="predict"),
]
urlpatterns = format_suffix_patterns(urlpatterns)
