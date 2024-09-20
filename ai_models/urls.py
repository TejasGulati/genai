from django.urls import path
from .views import (
    SustainabilityReportView,
    EnvironmentalImpactView,
    BusinessModelView,
    PredictionView,
    TextGenerationView,
    ImageGenerationView
)

urlpatterns = [
    path('sustainability-report/', SustainabilityReportView.as_view(), name='sustainability_report'),
    path('environmental-impact/', EnvironmentalImpactView.as_view(), name='environmental_impact'),
    path('business-model/', BusinessModelView.as_view(), name='business_model'),
    path('predict/', PredictionView.as_view(), name='predict'),
    path('generate-text/', TextGenerationView.as_view(), name='generate_text'),
    path('generate-image/', ImageGenerationView.as_view(), name='generate_image'),
]