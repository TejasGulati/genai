from rest_framework import serializers

class CompanyNameSerializer(serializers.Serializer):
    company_name = serializers.CharField(max_length=255)

class CompanyYearSerializer(serializers.Serializer):
    company = serializers.CharField(max_length=255)
    year = serializers.IntegerField()

class PredictionDataSerializer(serializers.Serializer):
    data = serializers.JSONField()
    dataset_key = serializers.CharField(max_length=255)

class TextPromptSerializer(serializers.Serializer):
    prompt = serializers.CharField()
    max_length = serializers.IntegerField(required=False, default=100)

class ImagePromptSerializer(serializers.Serializer):
    prompt = serializers.CharField()