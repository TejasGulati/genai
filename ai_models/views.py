from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .ml_model import initialized_model, TextGenerator, PredictiveAnalytics, EnvironmentalImpactAnalyzer, InnovativeBusinessModelGenerator, GenerativeImageCreator
from .serializers import CompanyNameSerializer, CompanyYearSerializer, PredictionDataSerializer, TextPromptSerializer, ImagePromptSerializer
import pandas as pd

class SustainabilityReportView(APIView):
    def post(self, request):
        serializer = CompanyNameSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        company_name = serializer.validated_data['company_name']
        report = initialized_model.generate_sustainability_report(company_name)
        return Response(report)

class EnvironmentalImpactView(APIView):
    def post(self, request):
        serializer = CompanyYearSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        company = serializer.validated_data['company']
        year = serializer.validated_data['year']
        analyzer = EnvironmentalImpactAnalyzer(initialized_model)
        impact = analyzer.analyze(company, year)
        return Response(impact)

class BusinessModelView(APIView):
    def post(self, request):
        serializer = CompanyYearSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        company = serializer.validated_data['company']
        year = serializer.validated_data['year']
        generator = InnovativeBusinessModelGenerator(initialized_model)
        business_model = generator.generate_business_model({'company': company, 'year': year})
        return Response(business_model)

class PredictionView(APIView):
    def post(self, request):
        serializer = PredictionDataSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data['data']
        dataset_key = serializer.validated_data['dataset_key']
        analytics = PredictiveAnalytics(initialized_model)
        predictions = analytics.predict(pd.DataFrame(data), dataset_key)
        return Response(predictions)

class TextGenerationView(APIView):
    def post(self, request):
        serializer = TextPromptSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        prompt = serializer.validated_data['prompt']
        max_length = serializer.validated_data['max_length']
        generator = TextGenerator(initialized_model)
        generated_text = generator.generate(prompt, max_length)
        return Response({"generated_text": generated_text})

class ImageGenerationView(APIView):
    def post(self, request):
        serializer = ImagePromptSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        prompt = serializer.validated_data['prompt']
        creator = GenerativeImageCreator()
        image_data = creator.create_image(prompt)
        return Response({"image_data": image_data})