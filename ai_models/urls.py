from django.urls import path
from .views import (
    TextGenerationView,
    ImageGenerationView,
    PredictiveAnalyticsView,
    EnvironmentalImpactView,
    ESGScoreView,
    InnovativeBusinessModelView,
    SustainabilityReportView,
    GeospatialAnalysisView,
    TimeSeriesForecastView,
    AIInsightsView
)

urlpatterns = [
    path('generate-text/', TextGenerationView.as_view(), name='generate_text'),
    path('generate-image/', ImageGenerationView.as_view(), name='generate_image'),
    path('predictive-analytics/', PredictiveAnalyticsView.as_view(), name='predictive_analytics'),
    path('environmental-impact/', EnvironmentalImpactView.as_view(), name='environmental_impact'),
    path('esg-score/', ESGScoreView.as_view(), name='esg_score'),
    path('business-model/', InnovativeBusinessModelView.as_view(), name='business_model'),
    path('sustainability-report/', SustainabilityReportView.as_view(), name='sustainability_report'),
    path('geospatial-analysis/', GeospatialAnalysisView.as_view(), name='geospatial_analysis'),
    path('time-series-forecast/', TimeSeriesForecastView.as_view(), name='time_series_forecast'),
    path('ai-insights/', AIInsightsView.as_view(), name='ai_insights'),
]