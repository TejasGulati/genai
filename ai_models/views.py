from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .ml_model import (
    EnhancedSustainabilityModel,
    TextGenerator,
    PredictiveAnalytics,
    EnvironmentalImpactAnalyzer,
    InnovativeBusinessModelGenerator
)
import json
import pandas as pd
from typing import Dict, Any

# Initialize the models
enhanced_model = EnhancedSustainabilityModel()
enhanced_model.load_data()
enhanced_model.preprocess_data()
enhanced_model.train_models()
enhanced_model.train_time_series_model()

text_generator = TextGenerator(enhanced_model)
predictive_analytics = PredictiveAnalytics(enhanced_model)
environmental_impact_analyzer = EnvironmentalImpactAnalyzer(enhanced_model)
business_model_generator = InnovativeBusinessModelGenerator(enhanced_model)

@csrf_exempt
@require_http_methods(["POST"])
def generate_sustainability_report(request):
    try:
        data = json.loads(request.body)
        company_name = data.get('company_name')
        
        if not company_name:
            # If no company name is provided, return a list of available companies
            available_companies = enhanced_model.get_available_companies(limit=100)
            return JsonResponse(available_companies)
        
        report = enhanced_model.generate_sustainability_report(company_name)
        
        if 'error' in report and 'available_companies' in report:
            # If the company wasn't found, return the error message with available companies
            return JsonResponse(report, status=404)
        
        return JsonResponse(report)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
@csrf_exempt
@require_http_methods(["POST"])
def generate_text(request):
    data = json.loads(request.body)
    prompt = data.get('prompt')
    max_length = data.get('max_length', 100)
    if not prompt:
        return JsonResponse({'error': 'Prompt is required'}, status=400)
    
    generated_text = text_generator.generate(prompt, max_length)
    return JsonResponse({'generated_text': generated_text})

@csrf_exempt
@require_http_methods(["POST"])
def make_prediction(request):
    data = json.loads(request.body)
    input_data = data.get('input_data')
    dataset_key = data.get('dataset_key')
    if not input_data or not dataset_key:
        return JsonResponse({'error': 'Input data and dataset key are required'}, status=400)
    
    df = pd.DataFrame(input_data)
    predictions = predictive_analytics.predict(df, dataset_key)
    return JsonResponse(predictions)

@csrf_exempt
@require_http_methods(["POST"])
def analyze_environmental_impact(request):
    data = json.loads(request.body)
    country = data.get('country')
    year = data.get('year')
    if not country or not year:
        return JsonResponse({'error': 'Country and year are required'}, status=400)
    
    analysis = environmental_impact_analyzer.analyze(country, year)
    return JsonResponse(analysis)

@csrf_exempt
@require_http_methods(["POST"])
def generate_business_model(request):
    data = json.loads(request.body)
    input_data = data.get('input_data')
    if not input_data:
        return JsonResponse({'error': 'Input data is required'}, status=400)
    
    business_model = business_model_generator.generate_business_model(input_data)
    return JsonResponse(business_model)

@csrf_exempt
@require_http_methods(["POST"])
def run_geospatial_analysis(request):
    analysis_result = enhanced_model.run_geospatial_analysis()
    return JsonResponse(analysis_result)

@csrf_exempt
@require_http_methods(["POST"])
def make_time_series_forecast(request):
    data = json.loads(request.body)
    country = data.get('country')
    indicator = data.get('indicator')
    if not country or not indicator:
        return JsonResponse({'error': 'Country and indicator are required'}, status=400)
    
    forecast = enhanced_model.make_time_series_forecast(country, indicator)
    return JsonResponse(forecast)

@csrf_exempt
@require_http_methods(["GET"])
def get_model_performance(request):
    dataset_key = request.GET.get('dataset_key')
    if not dataset_key:
        return JsonResponse({'error': 'Dataset key is required'}, status=400)
    
    cv_scores = enhanced_model.cv_scores.get(dataset_key, {})
    return JsonResponse({'cv_scores': cv_scores})

@csrf_exempt
@require_http_methods(["POST"])
def analyze_text_sentiment(request):
    data = json.loads(request.body)
    text = data.get('text')
    if not text:
        return JsonResponse({'error': 'Text is required'}, status=400)
    
    sentiment_analysis = business_model_generator.analyze_text(text)
    return JsonResponse(sentiment_analysis)

@csrf_exempt
@require_http_methods(["GET"])
def get_available_datasets(request):
    datasets = list(enhanced_model.data.keys())
    return JsonResponse({'available_datasets': datasets})

@csrf_exempt
@require_http_methods(["POST"])
def update_model(request):
    data = json.loads(request.body)
    dataset_key = data.get('dataset_key')
    new_data = data.get('new_data')
    if not dataset_key or not new_data:
        return JsonResponse({'error': 'Dataset key and new data are required'}, status=400)
    
    df = pd.DataFrame(new_data)
    enhanced_model.data[dataset_key] = pd.concat([enhanced_model.data[dataset_key], df], ignore_index=True)
    enhanced_model.preprocess_data()
    enhanced_model.train_models()
    return JsonResponse({'message': f'Model updated with new data for {dataset_key}'})

@csrf_exempt
@require_http_methods(["GET"])
def get_model_summary(request):
    dataset_key = request.GET.get('dataset_key')
    if not dataset_key:
        return JsonResponse({'error': 'Dataset key is required'}, status=400)
    
    model = enhanced_model.get_model(dataset_key)
    if model is None:
        return JsonResponse({'error': f'No model found for {dataset_key}'}, status=404)
    
    summary = {
        'model_type': str(type(model)),
        'feature_importance': enhanced_model._get_feature_importance(dataset_key)
    }
    return JsonResponse(summary)

@csrf_exempt
@require_http_methods(["POST"])
def generate_recommendations(request):
    data = json.loads(request.body)
    company_name = data.get('company_name')
    if not company_name:
        return JsonResponse({'error': 'Company name is required'}, status=400)
    
    report = enhanced_model.generate_sustainability_report(company_name)
    recommendations = report.get('recommendations', [])
    return JsonResponse({'recommendations': recommendations}) 