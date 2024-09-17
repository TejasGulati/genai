from django.urls import path
from .views import (
    SustainabilityReportView, TextGenerationView, 
    PredictiveAnalyticsView, 
    EnvironmentalImpactView, BusinessModelGeneratorView,
    TimeSeriesForecastView
)

urlpatterns = [
    path('sustainability-report/', SustainabilityReportView.as_view(), name='sustainability_report'),
    path('generate-text/', TextGenerationView.as_view(), name='generate_text'),
    path('predict/', PredictiveAnalyticsView.as_view(), name='predict'),
    path('environmental-impact/', EnvironmentalImpactView.as_view(), name='environmental_impact'),
    path('generate-business-model/', BusinessModelGeneratorView.as_view(), name='generate_business_model'),
    path('time-series-forecast/', TimeSeriesForecastView.as_view(), name='time_series_forecast'),
]