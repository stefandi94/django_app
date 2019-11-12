from django.urls import path, include
from rest_framework import routers
from rest_framework.urlpatterns import format_suffix_patterns

from app import views

# router = routers.DefaultRouter()
# router.register('app', views.TextView)
from app.views import Predict, Train

urlpatterns = [
    path('predict/', Predict.as_view(), name="predict"),
    path('train/', Train.as_view(), name="train"),
    path('predict_form/', views.text_predict, name="predict_form")
]


urlpatterns = format_suffix_patterns(urlpatterns)