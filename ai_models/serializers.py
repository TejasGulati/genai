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

from rest_framework import serializers

class CustomCompanyDataSerializer(serializers.Serializer):
    company_name = serializers.CharField(max_length=100)
    industry = serializers.CharField(max_length=100)
    year = serializers.IntegerField(min_value=2000, max_value=2100)
    ai_adoption_percentage = serializers.FloatField(min_value=0, max_value=100)
    primary_ai_application = serializers.CharField(max_length=100)
    esg_score = serializers.FloatField(min_value=0, max_value=100)
    primary_esg_impact = serializers.CharField(max_length=100)
    sustainable_growth_index = serializers.FloatField(min_value=0, max_value=1)
    innovation_index = serializers.FloatField(min_value=0, max_value=100)
    revenue_growth = serializers.FloatField()
    cost_reduction = serializers.FloatField()
    employee_satisfaction = serializers.FloatField(min_value=0, max_value=100)
    market_share_change = serializers.FloatField()