from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import (
    SustainabilityReportSerializer, TextGenerationSerializer, 
    PredictionSerializer, 
    EnvironmentalImpactSerializer, BusinessModelSerializer,
    TimeSeriesForecastSerializer
)
from .ml_model import EnhancedSustainabilityModel, TextGenerator,PredictiveAnalytics, EnvironmentalImpactAnalyzer, InnovativeBusinessModelGenerator
import pandas as pd

enhanced_model = EnhancedSustainabilityModel()
enhanced_model.load_data()
enhanced_model.preprocess_data()
enhanced_model.train_models()
enhanced_model.train_time_series_model()

text_generator = TextGenerator(enhanced_model)
predictive_analytics = PredictiveAnalytics(enhanced_model)
environmental_impact_analyzer = EnvironmentalImpactAnalyzer(enhanced_model)
business_model_generator = InnovativeBusinessModelGenerator(enhanced_model)

class SustainabilityReportView(APIView):
    def post(self, request):
        serializer = SustainabilityReportSerializer(data=request.data)
        if serializer.is_valid():
            company_name = serializer.validated_data['company_name']
            report = enhanced_model.generate_sustainability_report(company_name)
            return Response(report)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TextGenerationView(APIView):
    def post(self, request):
        serializer = TextGenerationSerializer(data=request.data)
        if serializer.is_valid():
            prompt = serializer.validated_data['prompt']
            csv_file = serializer.validated_data.get('csv_file')
            use_gpt2 = serializer.validated_data.get('use_gpt2', False)
            generated_text = text_generator.generate(prompt, csv_file, use_gpt2)
            return Response({"generated_text": generated_text})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PredictiveAnalyticsView(APIView):
    def post(self, request):
        serializer = PredictionSerializer(data=request.data)
        if serializer.is_valid():
            data = pd.DataFrame([serializer.validated_data['data']])
            dataset_key = serializer.validated_data['dataset_key']
            predictions = predictive_analytics.predict(data, dataset_key)
            return Response(predictions)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class EnvironmentalImpactView(APIView):
    def post(self, request):
        serializer = EnvironmentalImpactSerializer(data=request.data)
        if serializer.is_valid():
            country = serializer.validated_data['country']
            year = serializer.validated_data['year']
            impact_analysis = environmental_impact_analyzer.analyze(country, year)
            return Response(impact_analysis)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class BusinessModelGeneratorView(APIView):
    def post(self, request):
        serializer = BusinessModelSerializer(data=request.data)
        if serializer.is_valid():
            business_model = business_model_generator.generate_business_model(serializer.validated_data)
            return Response(business_model)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TimeSeriesForecastView(APIView):
    def post(self, request):
        serializer = TimeSeriesForecastSerializer(data=request.data)
        if serializer.is_valid():
            country = serializer.validated_data['country']
            indicator = serializer.validated_data['indicator']
            periods = serializer.validated_data.get('periods', 365)
            forecast = enhanced_model.make_time_series_forecast(country, indicator, periods)
            return Response(forecast)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)