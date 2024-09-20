from django.urls import path
from . import views

urlpatterns = [
    path('generate-sustainability-report/', views.generate_sustainability_report, name='generate_sustainability_report'),
    path('generate-text/', views.generate_text, name='generate_text'),
    path('make-prediction/', views.make_prediction, name='make_prediction'),
    path('analyze-environmental-impact/', views.analyze_environmental_impact, name='analyze_environmental_impact'),
    path('generate-business-model/', views.generate_business_model, name='generate_business_model'),
    path('run-geospatial-analysis/', views.run_geospatial_analysis, name='run_geospatial_analysis'),
    path('make-time-series-forecast/', views.make_time_series_forecast, name='make_time_series_forecast'),
    path('get-model-performance/', views.get_model_performance, name='get_model_performance'),
    path('analyze-text-sentiment/', views.analyze_text_sentiment, name='analyze_text_sentiment'),
    path('get-available-datasets/', views.get_available_datasets, name='get_available_datasets'),
    path('update-model/', views.update_model, name='update_model'),
    path('get-model-summary/', views.get_model_summary, name='get_model_summary'),
    path('generate-recommendations/', views.generate_recommendations, name='generate_recommendations'),
]