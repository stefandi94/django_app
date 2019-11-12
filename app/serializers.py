from rest_framework import serializers

from app.models import TrainModel, PredictModel


# class TextSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = TextModel
#         fields = '__all__'
# #
# class TextSerializer(serializers.Serializer):
#     text = serializers.CharField(required=True)

class TrainSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainModel
        fields = '__all__'


class PredictSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictModel
        fields = '__all__'
