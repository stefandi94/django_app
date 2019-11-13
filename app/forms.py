from django import forms

from .models import TrainModel, PredictModel


class TrainForm(forms.ModelForm):
    class Meta:
        model = TrainModel
        fields = '__all__'


class PredictForm(forms.ModelForm):
    class Meta:
        model = PredictModel
        fields = '__all__'
        widgets = {
            'text': forms.Textarea(attrs={'placeholder': 'Enter comment here'}),
            'load_path': forms.TextInput(attrs={'size': 50})
        }
