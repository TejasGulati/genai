import os
from rest_framework import views, status
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import logging
import json
import google.generativeai as genai
from .ml_model import EnhancedSustainabilityModel, TextGenerator, ImageGenerator, PredictiveAnalytics, EnvironmentalImpactAnalyzer, ESGScoreCalculator, InnovativeBusinessModelGenerator
from .serializers import (
    TextGenerationSerializer, 
    ImageGenerationSerializer, 
    PredictiveAnalyticsSerializer, 
    EnvironmentalImpactSerializer,
    ESGScoreSerializer,
    BusinessModelSerializer,
    SustainabilityReportSerializer,
    GeospatialAnalysisSerializer
)

logger = logging.getLogger(__name__)

# Initialize Gemini and EnhancedSustainabilityModel
genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')
sustainability_model = EnhancedSustainabilityModel()
sustainability_model.load_data()
sustainability_model.preprocess_data()
sustainability_model.engineer_features()
sustainability_model.train_models()
sustainability_model.train_time_series_model()

text_generator = TextGenerator(sustainability_model)
image_generator = ImageGenerator(sustainability_model)
predictive_analytics = PredictiveAnalytics(sustainability_model)
environmental_impact_analyzer = EnvironmentalImpactAnalyzer(sustainability_model)
esg_score_calculator = ESGScoreCalculator(sustainability_model)
business_model_generator = InnovativeBusinessModelGenerator(sustainability_model)

@method_decorator(csrf_exempt, name='dispatch')
class TextGenerationView(views.APIView):
    def post(self, request):
        serializer = TextGenerationSerializer(data=request.data)
        if serializer.is_valid():
            try:
                generated_text = text_generator.generate(
                    serializer.validated_data['prompt'],
                    serializer.validated_data.get('csv_file'),
                    use_gpt2=serializer.validated_data.get('use_gpt2', False)
                )
                return Response({'generated_text': generated_text}, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Error in text generation: {str(e)}")
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name='dispatch')
class ImageGenerationView(views.APIView):
    def post(self, request):
        serializer = ImageGenerationSerializer(data=request.data)
        if serializer.is_valid():
            try:
                image_url = image_generator.generate(serializer.validated_data['prompt'])
                return Response({'image_url': image_url}, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Error in image generation: {str(e)}")
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name='dispatch')
class PredictiveAnalyticsView(views.APIView):
    def post(self, request):
        serializer = PredictiveAnalyticsSerializer(data=request.data)
        if serializer.is_valid():
            try:
                predictions = predictive_analytics.predict(
                    serializer.validated_data['data'],
                    serializer.validated_data['dataset_key']
                )
                return Response(predictions, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Error in predictive analytics: {str(e)}")
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name='dispatch')
class EnvironmentalImpactView(views.APIView):
    def post(self, request):
        serializer = EnvironmentalImpactSerializer(data=request.data)
        if serializer.is_valid():
            try:
                impact_analysis = environmental_impact_analyzer.analyze(
                    serializer.validated_data['country'],
                    serializer.validated_data['year']
                )
                return Response(impact_analysis, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Error in environmental impact analysis: {str(e)}")
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name='dispatch')
class ESGScoreView(views.APIView):
    def post(self, request):
        serializer = ESGScoreSerializer(data=request.data)
        if serializer.is_valid():
            try:
                esg_score = esg_score_calculator.calculate_esg_score(serializer.validated_data['company_data'])
                return Response(esg_score, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Error in ESG score calculation: {str(e)}")
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name='dispatch')
class InnovativeBusinessModelView(views.APIView):
    def post(self, request):
        serializer = BusinessModelSerializer(data=request.data)
        if serializer.is_valid():
            try:
                business_model = business_model_generator.generate_business_model(serializer.validated_data)
                return Response(business_model, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Error in business model generation: {str(e)}")
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name='dispatch')
class SustainabilityReportView(views.APIView):
    def post(self, request):
        serializer = SustainabilityReportSerializer(data=request.data)
        if serializer.is_valid():
            try:
                report = sustainability_model.generate_sustainability_report(serializer.validated_data['company_name'])
                return Response(report, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Error generating sustainability report: {str(e)}")
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name='dispatch')
class GeospatialAnalysisView(views.APIView):
    def post(self, request):
        serializer = GeospatialAnalysisSerializer(data=request.data)
        if serializer.is_valid():
            try:
                analysis_results = sustainability_model.run_geospatial_analysis()
                return Response(analysis_results, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Error in geospatial analysis: {str(e)}")
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name='dispatch')
class TimeSeriesForecastView(views.APIView):
    def post(self, request):
        try:
            periods = request.data.get('periods', 365)
            forecast = sustainability_model.make_time_series_forecast(periods)
            return Response(forecast, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in time series forecast: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class AIInsightsView(views.APIView):
    def post(self, request):
        try:
            data = request.data
            if not data:
                return Response({'error': 'No data provided'}, status=status.HTTP_400_BAD_REQUEST)

            prompt = f"""
            Analyze the following business data and provide insights:

            {json.dumps(data, indent=2)}

            Generate a JSON object with the following structure:
            {{
                "key_insights": [
                    {{"insight": "Description of key insight", "importance": "High/Medium/Low"}}
                ],
                "recommendations": [
                    {{"action": "Recommended action", "expected_impact": "Description of expected impact"}}
                ],
                "potential_risks": [
                    {{"risk": "Description of potential risk", "mitigation": "Suggested mitigation strategy"}}
                ],
                "opportunities": [
                    {{"opportunity": "Description of opportunity", "potential_benefit": "Description of potential benefit"}}
                ]
            }}
            Ensure all sections are filled with detailed, relevant content based on the provided data.
            """

            response = model.generate_content(prompt)
            insights = json.loads(response.text)

            return Response(insights, status=status.HTTP_200_OK)
        except json.JSONDecodeError:
            logger.error("Error decoding AI response")
            return Response({'error': 'Error processing AI insights'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Error in AI insights generation: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)