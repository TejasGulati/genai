from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .ml_model import initialized_model, TextGenerator, PredictiveAnalytics, EnvironmentalImpactAnalyzer, InnovativeBusinessModelGenerator, GenerativeImageCreator
import pandas as pd

class SustainabilityReportView(APIView):
    def post(self, request):
        company_name = request.data.get('company_name')
        if not company_name:
            return Response({"error": "Company name is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        report = initialized_model.generate_sustainability_report(company_name)
        return Response(report)

class AIInsightsView(APIView):
    def post(self, request):
        company_name = request.data.get('company_name')
        if not company_name:
            return Response({"error": "Company name is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        insights = initialized_model.generate_ai_driven_insights(company_name)
        return Response(insights)

class EnvironmentalImpactView(APIView):
    def post(self, request):
        company = request.data.get('company')
        year = request.data.get('year')
        if not company or not year:
            return Response({"error": "Company and year are required"}, status=status.HTTP_400_BAD_REQUEST)
        
        analyzer = EnvironmentalImpactAnalyzer(initialized_model)
        impact = analyzer.analyze(company, year)
        return Response(impact)

class BusinessModelView(APIView):
    def post(self, request):
        company = request.data.get('company')
        year = request.data.get('year')
        if not company or not year:
            return Response({"error": "Company and year are required"}, status=status.HTTP_400_BAD_REQUEST)
        
        generator = InnovativeBusinessModelGenerator(initialized_model)
        business_model = generator.generate_business_model({'company': company, 'year': year})
        return Response(business_model)

class PredictionView(APIView):
    def post(self, request):
        data = request.data.get('data')
        dataset_key = request.data.get('dataset_key')
        if not data or not dataset_key:
            return Response({"error": "Data and dataset_key are required"}, status=status.HTTP_400_BAD_REQUEST)
        
        analytics = PredictiveAnalytics(initialized_model)
        predictions = analytics.predict(pd.DataFrame(data), dataset_key)
        return Response(predictions)

class TextGenerationView(APIView):
    def post(self, request):
        prompt = request.data.get('prompt')
        max_length = request.data.get('max_length', 100)
        if not prompt:
            return Response({"error": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        generator = TextGenerator(initialized_model)
        generated_text = generator.generate(prompt, max_length)
        return Response({"generated_text": generated_text})

class ImageGenerationView(APIView):
    def post(self, request):
        prompt = request.data.get('prompt')
        if not prompt:
            return Response({"error": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        creator = GenerativeImageCreator()
        image_data = creator.create_image(prompt)
        return Response({"image_data": image_data})