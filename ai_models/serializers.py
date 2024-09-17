from rest_framework import serializers

class SustainabilityReportSerializer(serializers.Serializer):
    company_name = serializers.CharField()
    industry = serializers.CharField()
    country = serializers.CharField()
    year_founded = serializers.IntegerField()
    size_range = serializers.CharField()
    esg_scores = serializers.DictField()
    sustainability_prediction = serializers.FloatField()
    key_factors = serializers.DictField()
    recommendations = serializers.ListField(child=serializers.CharField())
    country_metrics = serializers.DictField()
    sdg_indicator = serializers.DictField(required=False)

class TextGenerationSerializer(serializers.Serializer):
    prompt = serializers.CharField()
    csv_file = serializers.CharField(required=False, allow_null=True)
    use_gpt2 = serializers.BooleanField(default=False)

class PredictionSerializer(serializers.Serializer):
    data = serializers.DictField()
    dataset_key = serializers.CharField()

class EnvironmentalImpactSerializer(serializers.Serializer):
    country = serializers.CharField()
    year = serializers.IntegerField()

class BusinessModelSerializer(serializers.Serializer):
    industry = serializers.CharField()
    target_market = serializers.CharField()
    key_resources = serializers.ListField(child=serializers.CharField())
    description = serializers.CharField(required=False)
    country = serializers.CharField(required=False)

class TimeSeriesForecastSerializer(serializers.Serializer):
    country = serializers.CharField()
    indicator = serializers.CharField()
    periods = serializers.IntegerField(default=365)