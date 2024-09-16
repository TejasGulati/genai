from rest_framework import serializers

class TextGenerationSerializer(serializers.Serializer):
    prompt = serializers.CharField(max_length=1000)
    csv_file = serializers.ChoiceField(choices=['air_quality', 'companies', 'energy', 'esg_score', 'innovative_startups', 'sdg_indicator', 'world_bank'], required=False)
    use_gpt2 = serializers.BooleanField(default=False)

class ImageGenerationSerializer(serializers.Serializer):
    prompt = serializers.CharField(max_length=1000)

class PredictiveAnalyticsSerializer(serializers.Serializer):
    data = serializers.DictField()
    dataset_key = serializers.ChoiceField(choices=['air_quality', 'companies', 'energy', 'esg_score', 'innovative_startups', 'sdg_indicator', 'world_bank'])

class EnvironmentalImpactSerializer(serializers.Serializer):
    country = serializers.CharField(max_length=100)
    year = serializers.IntegerField()

class ESGScoreSerializer(serializers.Serializer):
    company_data = serializers.DictField()

class BusinessModelSerializer(serializers.Serializer):
    industry = serializers.CharField(max_length=100)
    target_market = serializers.CharField(max_length=100)
    key_resources = serializers.ListField(child=serializers.CharField(max_length=100))
    description = serializers.CharField(max_length=1000, required=False)
    country = serializers.CharField(max_length=100, required=False)

class SustainabilityReportSerializer(serializers.Serializer):
    company_name = serializers.CharField(max_length=100)

class GeospatialAnalysisSerializer(serializers.Serializer):
    pass  # No input needed for geospatial analysis

class TimeSeriesForecastSerializer(serializers.Serializer):
    periods = serializers.IntegerField(default=365)

class ModelTrainingSerializer(serializers.Serializer):
    pass  # No input needed for model training

class DataLoadingSerializer(serializers.Serializer):
    pass  # No input needed for data loading

class FeatureEngineeringSerializer(serializers.Serializer):
    pass  # No input needed for feature engineering

class ModelExplanationSerializer(serializers.Serializer):
    key = serializers.CharField(max_length=100)

class CombinedAnalysisSerializer(serializers.Serializer):
    company_name = serializers.CharField(max_length=100)
    industry = serializers.CharField(max_length=100)
    country = serializers.CharField(max_length=100)
    year = serializers.IntegerField()