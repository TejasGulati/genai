from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from .ml_model import initialized_model, TextGenerator, PredictiveAnalytics, EnvironmentalImpactAnalyzer, InnovativeBusinessModelGenerator, GenerativeImageCreator
from .serializers import CompanyNameSerializer, CompanyYearSerializer, PredictionDataSerializer, TextPromptSerializer, ImagePromptSerializer
import pandas as pd
import google.generativeai as genai
from django.conf import settings
import json
import logging
import os
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .ml_model import initialized_model, EnvironmentalImpactAnalyzer, InnovativeBusinessModelGenerator
from .serializers import CompanyNameSerializer, CompanyYearSerializer, CustomCompanyDataSerializer

logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
ai_model = genai.GenerativeModel('gemini-pro')

class AIEnhancedView(APIView):
    permission_classes = [IsAuthenticated]
    
    def _generate_ai_content(self, prompt, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                response = ai_model.generate_content(prompt)
                parsed_content = self._parse_ai_response(response.text)
                if parsed_content:
                    return parsed_content
            except Exception as e:
                logger.warning(f"AI content generation attempt {attempt + 1} failed: {str(e)}")
        
        logger.error(f"All {max_attempts} attempts to generate AI content failed.")
        return None

    def _parse_ai_response(self, response_text):
        try:
            cleaned_text = response_text.strip().strip('`')
            
            if cleaned_text.lower().startswith('json'):
                cleaned_text = cleaned_text[4:].lstrip()
            
            parsed_json = json.loads(cleaned_text)
            
            return self._clean_and_structure_json(parsed_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response as JSON. Attempting to structure the raw text.")
            return self._structure_raw_text(cleaned_text)

    def _clean_and_structure_json(self, data):
        if isinstance(data, dict):
            return {self._clean_key(k): self._clean_and_structure_json(v) 
                    for k, v in data.items() if v is not None and v != ""}
        elif isinstance(data, list):
            return [self._clean_and_structure_json(item) 
                    for item in data if item is not None and item != ""]
        elif isinstance(data, str):
            return self._clean_text(data)
        else:
            return data

    @staticmethod
    def _clean_key(key):
        cleaned_key = ''.join(c.lower() if c.isalnum() else '_' for c in key)
        return '_'.join(word for word in cleaned_key.split('_') if word)

    def _clean_text(self, text):
        cleaned = text.replace('\\n', ' ').replace('\\', '').strip()
        return ' '.join(cleaned.split())

    def _structure_raw_text(self, text):
        lines = text.split('\n')
        structured_data = {}
        current_key = None
        current_list = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.endswith(':'):
                current_key = self._clean_key(line[:-1])
                structured_data[current_key] = {}
                current_list = None
            elif line.startswith('- '):
                if current_list is None:
                    current_list = []
                    structured_data[current_key] = current_list
                current_list.append(line[2:])
            elif ':' in line:
                key, value = line.split(':', 1)
                if current_key:
                    structured_data[current_key][self._clean_key(key)] = value.strip()
                else:
                    structured_data[self._clean_key(key)] = value.strip()
            else:
                if current_list is not None:
                    current_list[-1] += ' ' + line
                elif current_key:
                    last_subkey = list(structured_data[current_key].keys())[-1]
                    structured_data[current_key][last_subkey] += ' ' + line
                else:
                    structured_data[current_key] += ' ' + line
        return structured_data

    def _serialize_json(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

class SustainabilityReportView(AIEnhancedView):
    def post(self, request):
        if 'custom_data' in request.data:
            serializer = CustomCompanyDataSerializer(data=request.data['custom_data'])
        else:
            serializer = CompanyNameSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        if 'custom_data' in request.data:
            report = self._generate_custom_sustainability_report(serializer.validated_data)
        else:
            company_name = serializer.validated_data['company_name']
            report = initialized_model.generate_sustainability_report(company_name)
        
        ai_insights = self._generate_ai_insights(report['company_name'], report)
        
        return Response({
            "report": report,
            "ai_insights": ai_insights
        })

    def _generate_custom_sustainability_report(self, data):
        report = {
            "company_name": data['company_name'],
            "industry": data['industry'],
            "year": data['year'],
            "ai_adoption_percentage": data['ai_adoption_percentage'],
            "primary_ai_application": data['primary_ai_application'],
            "esg_score": data['esg_score'],
            "primary_esg_impact": data['primary_esg_impact'],
            "sustainable_growth_index": data['sustainable_growth_index'],
            "innovation_index": data['innovation_index'],
            "predictions": {
                "ai_esg_alignment": data['esg_score'],
                "ai_impact": data['cost_reduction'],
                "gen_ai_business": data['sustainable_growth_index']
            },
            "recommendations": self._generate_recommendations(data)
        }
        return report

    def _generate_recommendations(self, data):
        recommendations = []
        if data['ai_adoption_percentage'] < 50:
            recommendations.append("Consider increasing AI adoption to improve overall performance.")
        if data['esg_score'] < 70:
            recommendations.append(f"Focus on improving {data['primary_esg_impact']} to boost ESG performance.")
        if data['sustainable_growth_index'] < 0.5:
            recommendations.append("Develop strategies to enhance sustainable growth.")
        return recommendations

    def _generate_ai_insights(self, company_name, report):
        try:
            prompt = f"""
            Analyze the following sustainability report for {company_name}:

            {json.dumps(report, indent=2, default=self._serialize_json)}

            Please provide:
            1. Key strengths in the company's sustainability efforts.
            2. Areas for improvement and potential risks.
            3. Innovative suggestions for enhancing sustainability practices.
            4. Comparison with industry benchmarks (if available).
            5. Long-term sustainability outlook for the company.

            Format your response as a structured JSON object with appropriate keys for each section.
            """

            return self._generate_ai_content(prompt)
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return None

class EnvironmentalImpactView(AIEnhancedView):
    def post(self, request):
        if 'custom_data' in request.data:
            serializer = CustomCompanyDataSerializer(data=request.data['custom_data'])
        else:
            serializer = CompanyYearSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        if 'custom_data' in request.data:
            impact = self._analyze_custom_environmental_impact(serializer.validated_data)
        else:
            company = serializer.validated_data['company']
            year = serializer.validated_data['year']
            analyzer = EnvironmentalImpactAnalyzer(initialized_model)
            impact = analyzer.analyze(company, year)
        
        ai_analysis = self._generate_ai_analysis(impact['company'], impact['year'], impact)
        
        return Response({
            "impact": impact,
            "ai_analysis": ai_analysis
        })

    def _analyze_custom_environmental_impact(self, data):
        impact = {
            "company": data['company_name'],
            "year": data['year'],
            "impact_score": data['esg_score'],
            "ai_adoption_percentage": data['ai_adoption_percentage'],
            "primary_ai_application": data['primary_ai_application'],
            "esg_score": data['esg_score'],
            "primary_esg_impact": data['primary_esg_impact'],
            "recommendations": self._generate_environmental_recommendations(data)
        }
        return impact

    def _generate_environmental_recommendations(self, data):
        recommendations = []
        if data['esg_score'] < 70:
            recommendations.append(f"Increase focus on {data['primary_esg_impact']} to improve overall environmental impact.")
        if data['ai_adoption_percentage'] < 50:
            recommendations.append(f"Consider expanding use of AI in {data['primary_ai_application']} to drive efficiency and sustainability.")
        if data['sustainable_growth_index'] < 0.5:
            recommendations.append("Develop strategies to improve sustainable growth, possibly by leveraging AI technologies.")
        return recommendations

    def _generate_ai_analysis(self, company, year, impact):
        try:
            prompt = f"""
            Analyze the environmental impact of {company} for the year {year}:

            {json.dumps(impact, indent=2, default=self._serialize_json)}

            Please provide:
            1. A detailed interpretation of the environmental impact data.
            2. Potential short-term and long-term consequences of this impact.
            3. Recommendations for mitigating negative impacts.
            4. Innovative strategies for improving environmental performance.
            5. Comparison with industry standards and best practices.

            Format your response as a structured JSON object with appropriate keys for each section.
            """

            return self._generate_ai_content(prompt)
        except Exception as e:
            logger.error(f"Error generating AI analysis: {str(e)}")
            return None
from rest_framework.response import Response
from rest_framework import status
import json
import logging

logger = logging.getLogger(__name__)

class BusinessModelView(AIEnhancedView):
    def post(self, request):
        if 'custom_data' in request.data:
            serializer = CustomCompanyDataSerializer(data=request.data['custom_data'])
        else:
            serializer = CompanyYearSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            if 'custom_data' in request.data:
                business_model = self._generate_custom_business_model(serializer.validated_data)
            else:
                company = serializer.validated_data['company']
                year = serializer.validated_data['year']
                generator = InnovativeBusinessModelGenerator(initialized_model)
                business_model = generator.generate_business_model({'company': company, 'year': year})
            
            company = business_model.get('company', 'Unknown Company')
            year = business_model.get('year', 'Unknown Year')
            
            ai_enhancements = self._generate_ai_enhancements(company, year, business_model)
            
            return Response({
                "business_model": business_model,
                "ai_enhancements": ai_enhancements
            })
        except Exception as e:
            logger.error(f"Error in BusinessModelView: {str(e)}")
            return Response({"error": "An unexpected error occurred"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _generate_custom_business_model(self, data):
        business_model = {
            "company": data['company_name'],
            "year": data['year'],
            "industry": data['industry'],
            "ai_adoption_percentage": data['ai_adoption_percentage'],
            "primary_ai_application": data['primary_ai_application'],
            "esg_score": data['esg_score'],
            "primary_esg_impact": data['primary_esg_impact'],
            "sustainable_growth_index": data['sustainable_growth_index'],
            "innovation_score": data['innovation_index'],
            "market_insights": {
                "revenue_growth": data['revenue_growth'],
                "cost_reduction": data['cost_reduction'],
                "employee_satisfaction": data['employee_satisfaction'],
                "market_share_change": data['market_share_change']
            },
            "sustainability_metrics": {
                "esg_score": data['esg_score'],
                "sustainable_growth_index": data['sustainable_growth_index']
            },
            "innovation_score": data['innovation_index'],
            "future_trends": self._generate_future_trends(data)
        }
        return business_model

    def _generate_future_trends(self, data):
        return {
            "sustainable_growth_forecast": [
                data['sustainable_growth_index'] * (1 + 0.05 * i) for i in range(1, 6)
            ],
            "forecast_dates": [data['year'] + i for i in range(1, 6)]
        }

    def _generate_ai_enhancements(self, company, year, business_model):
        try:
            prompt = f"""
            Enhance the following innovative business model for {company} in {year}:

            {json.dumps(business_model, indent=2, default=self._serialize_json)}

            Please provide:
            1. Additional innovative elements to incorporate into the business model.
            2. Potential challenges and strategies to overcome them.
            3. Key performance indicators (KPIs) to measure the success of this model.
            4. Integration strategies with existing business operations.
            5. Long-term sustainability and scalability assessment.

            Format your response as a structured JSON object with appropriate keys for each section.
            """

            return self._generate_ai_content(prompt)
        except Exception as e:
            logger.error(f"Error generating AI enhancements: {str(e)}")
            return None

    def _serialize_json(self, obj):
        """Custom JSON serializer for objects not serializable by default json code"""
        return str(obj)
    
class PredictionView(AIEnhancedView):
    def post(self, request):
        serializer = PredictionDataSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data['data']
        dataset_key = serializer.validated_data['dataset_key']
        analytics = PredictiveAnalytics(initialized_model)
        predictions = analytics.predict(pd.DataFrame(data), dataset_key)
        
        ai_insights = self._generate_ai_insights(data, dataset_key, predictions)
        
        return Response({
            "predictions": predictions,
            "ai_insights": ai_insights
        })

    def _generate_ai_insights(self, data, dataset_key, predictions):
        try:
            prompt = f"""
            Analyze the following predictions for the {dataset_key} dataset:

            Input Data:
            {json.dumps(data, indent=2, default=self._serialize_json)}

            Predictions:
            {json.dumps(predictions, indent=2, default=self._serialize_json)}

            Please provide:
            1. Interpretation of the prediction results.
            2. Key factors influencing these predictions.
            3. Potential implications of these predictions.
            4. Recommendations based on the predicted outcomes.
            5. Suggestions for further analysis or data collection.

            Format your response as a structured JSON object with appropriate keys for each section.
            """

            return self._generate_ai_content(prompt)
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return None

class TextGenerationView(AIEnhancedView):
    def post(self, request):
        serializer = TextPromptSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        prompt = serializer.validated_data['prompt']
        max_length = serializer.validated_data['max_length']
        
        generated_text = self._generate_enhanced_text(prompt, max_length)
        
        return Response({
            "generated_text": generated_text
        })

    def _generate_enhanced_text(self, prompt, max_length):
        try:
            ai_prompt = f"""
            Generate an enhanced version of the following text, expanding on the ideas and adding depth:

            Original Prompt: {prompt}

            Please provide:
            1. An expanded version of the text (approximately {max_length} words).
            2. Key themes or topics covered in the generated text.
            3. Potential applications or use cases for this generated content.

            Format your response as a structured JSON object with appropriate keys for each section.
            """

            return self._generate_ai_content(ai_prompt)
        except Exception as e:
            logger.error(f"Error generating enhanced text: {str(e)}")
            return None
        
import os
import base64
import shutil
from django.conf import settings
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
import logging
from urllib.parse import urljoin, urlparse
import requests

logger = logging.getLogger(__name__)

class ImageGenerationView(AIEnhancedView):
    def post(self, request):
        try:
            serializer = ImagePromptSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            prompt = serializer.validated_data['prompt']
            creator = GenerativeImageCreator()
            image_data = creator.create_image(prompt)
            
            logger.info(f"Received image_data: {type(image_data)}")
            if isinstance(image_data, str):
                logger.info(f"Image data content: {image_data[:100]}...")  # Log first 100 chars
            
            image_path = self._save_image(image_data)
            logger.info(f"Image saved successfully at {image_path}")
            
            # Generate full URL for the image
            full_image_url = self._get_full_url(request, image_path)
            
            ai_description = self._generate_ai_description(prompt, full_image_url)
            
            return Response({
                "image_url": full_image_url,
                "ai_description": ai_description
            })
        
        except Exception as e:
            logger.exception(f"Error in image generation: {str(e)}")
            return Response({"error": "An unexpected error occurred"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _save_image(self, image_data):
        try:
            filename = f"image_{timezone.now().strftime('%Y%m%d%H%M%S')}.png"
            relative_path = os.path.join('generated_images', filename)
            full_path = os.path.join(settings.MEDIA_ROOT, relative_path)
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            if isinstance(image_data, str):
                if image_data.startswith('http'):
                    # It's a URL, download the image
                    response = requests.get(image_data)
                    response.raise_for_status()
                    image_data = response.content
                elif image_data.startswith('data:image'):
                    # It's a base64 encoded image
                    image_data = base64.b64decode(image_data.split(',')[1])
                elif image_data.startswith('Image saved at '):
                    # It's a file path returned by the AI, but the file doesn't exist yet
                    # We'll assume the AI has created the file and we just need to move it
                    ai_generated_path = image_data.replace('Image saved at ', '').strip()
                    if os.path.exists(ai_generated_path):
                        shutil.move(ai_generated_path, full_path)
                        return os.path.join(settings.MEDIA_URL, relative_path)
                    else:
                        raise FileNotFoundError(f"AI reported saving image at {ai_generated_path}, but file not found")
                else:
                    # Assume it's a file path
                    with open(image_data, 'rb') as f:
                        image_data = f.read()
            
            if not isinstance(image_data, bytes):
                raise ValueError(f"Unexpected image_data type: {type(image_data)}")
            
            with open(full_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"Image saved at {full_path}")
            return os.path.join(settings.MEDIA_URL, relative_path)
        
        except Exception as e:
            logger.exception(f"Error saving image: {str(e)}")
            raise

    def _get_full_url(self, request, path):
        return request.build_absolute_uri(path)

    def _generate_ai_description(self, prompt, image_url):
        try:
            ai_prompt = f"""
            Describe and analyze the following generated image:

            Image Generation Prompt: {prompt}
            Image URL: {image_url}

            Please provide:
            1. A detailed description of the generated image.
            2. Analysis of how well the image matches the given prompt.
            3. Potential improvements or variations for future image generation.
            4. Possible use cases or applications for this generated image.

            Format your response as a structured JSON object with appropriate keys for each section.
            """

            return self._generate_ai_content(ai_prompt)
        except Exception as e:
            logger.error(f"Error generating AI description: {str(e)}")
            return None
