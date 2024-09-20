import os
import logging
from prophet import Prophet
from typing import Dict, List, Any
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from transformers import AutoTokenizer, AutoModel
import warnings
import google.generativeai as genai
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
stability_api = client.StabilityInference(key=os.getenv('STABILITY_KEY'), verbose=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.data = {}
        self.preprocessors = {}
        self.feature_names = {}
        self.target_variables = {}

    def load_data(self):
        csv_files = {
            'ai_esg_alignment': 'ai_esg_alignment.csv',
            'ai_impact': 'ai_impact_on_traditional_industries.csv',
            'gen_ai_business': 'generative_ai_business_models.csv'
        }
        
        for key, file in csv_files.items():
            try:
                file_path = os.path.join(os.path.dirname(__file__), '..', 'data', file)
                self.data[key] = pd.read_csv(file_path)
                logger.info(f"Loaded {key} dataset. Shape: {self.data[key].shape}")
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
                self.data[key] = None

    def identify_target_variable(self, key):
        target_mapping = {
            'ai_esg_alignment': 'esg_performance_score',
            'ai_impact': 'process_efficiency_improvement',
            'gen_ai_business': 'sustainable_growth_index'
        }
        
        if target_mapping[key] in self.data[key].columns:
            self.target_variables[key] = target_mapping[key]
        else:
            logger.warning(f"Target variable {target_mapping[key]} not found in {key} dataset. Using fallback.")
            numerical_columns = self.data[key].select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_columns) > 0:
                self.target_variables[key] = numerical_columns[-1]  # Use the last numerical column as target
                logger.info(f"Using {self.target_variables[key]} as target variable for {key} dataset.")
            else:
                logger.warning(f"No suitable target variable found for {key} dataset.")
                self.target_variables[key] = None

    def prepare_preprocessor(self, key):
        if self.data[key] is None or self.data[key].empty:
            logger.warning(f"No data available for {key}. Skipping preprocessor preparation.")
            return

        columns = self.data[key].columns
        numeric_features = self.data[key].select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.data[key].select_dtypes(include=['object']).columns

        # Remove the target variable from features if it exists
        if self.target_variables[key] in numeric_features:
            numeric_features = numeric_features.drop(self.target_variables[key])
        elif self.target_variables[key] in categorical_features:
            categorical_features = categorical_features.drop(self.target_variables[key])

        # Remove 'id', 'company', 'industry', and 'year' from features
        columns_to_drop = ['id', 'company', 'industry', 'year']
        numeric_features = numeric_features.drop([col for col in columns_to_drop if col in numeric_features])
        categorical_features = categorical_features.drop([col for col in columns_to_drop if col in categorical_features])

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessors[key] = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        self.feature_names[key] = list(numeric_features) + list(categorical_features)

    def get_features_and_target(self, key):
        if self.data[key] is None or self.data[key].empty:
            logger.warning(f"No data available for {key}. Skipping feature and target extraction.")
            return None, None

        if self.target_variables[key] is None:
            logger.warning(f"No target variable defined for {key}. Skipping feature and target extraction.")
            return None, None
        
        columns_to_drop = ['id', 'company', 'industry', 'year']
        columns_to_drop = [col for col in columns_to_drop if col in self.data[key].columns]
        columns_to_drop.append(self.target_variables[key])
        
        X = self.data[key].drop(columns=columns_to_drop, errors='ignore')
        y = self.data[key][self.target_variables[key]]
        return X, y

    def preprocess_data(self):
        for key in self.data:
            if self.data[key] is not None and not self.data[key].empty:
                self.handle_missing_values(key)
                self.identify_target_variable(key)
                self.prepare_preprocessor(key)
            else:
                logger.warning(f"No data available for {key}. Skipping preprocessing.")

    def handle_missing_values(self, key):
        for column in self.data[key].columns:
            if self.data[key][column].dtype in ['int64', 'float64']:
                self.data[key][column] = self.data[key][column].fillna(self.data[key][column].mean())
            else:
                self.data[key][column] = self.data[key][column].fillna(self.data[key][column].mode()[0])
class EnhancedSustainabilityModel:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.models = {}
        self.prophet_model = None
        self.genai_model = genai.GenerativeModel('gemini-pro')
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.data_processor.load_data()
        for key, data in self.data_processor.data.items():
            if data is not None:
                self.logger.info(f"Loaded {key} dataset. Shape: {data.shape}")
            else:
                self.logger.warning(f"Failed to load {key} dataset")

    def preprocess_data(self):
        self.data_processor.preprocess_data()

    def train_models(self):
        for key in self.data_processor.data:
            X, y = self.data_processor.get_features_and_target(key)
            if X is None or y is None:
                self.logger.warning(f"Skipping {key} dataset due to missing data or target variable")
                continue

            self.logger.info(f"Starting model training for {key} dataset")

            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                preprocessor = self.data_processor.preprocessors[key]
                
                X_train = X_train[self.data_processor.feature_names[key]]
                X_test = X_test[self.data_processor.feature_names[key]]
                
                X_train_preprocessed = preprocessor.fit_transform(X_train)
                X_test_preprocessed = preprocessor.transform(X_test)

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_preprocessed, y_train)

                self.models[key] = model

                y_pred = model.predict(X_test_preprocessed)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                self.logger.info(f"{key} - Test set MSE: {mse:.4f}, R2: {r2:.4f}")
            except Exception as e:
                self.logger.error(f"Error training model for {key} dataset: {str(e)}")

    def train_time_series_model(self):
        if 'gen_ai_business' in self.data_processor.data and self.data_processor.data['gen_ai_business'] is not None:
            try:
                data = self.data_processor.data['gen_ai_business']
                data['ds'] = pd.to_datetime(data['year'], format='%Y')
                data['y'] = data['sustainable_growth_index']

                self.prophet_model = Prophet()
                self.prophet_model.fit(data[['ds', 'y']])
                self.logger.info("Successfully trained Prophet time series model")
            except Exception as e:
                self.logger.error(f"Error training Prophet model: {str(e)}")
                self.prophet_model = None
        else:
            self.logger.warning("Unable to train time series model: 'gen_ai_business' data is missing or empty")
            self.prophet_model = None

    def generate_sustainability_report(self, company_name):
        try:
            company_data = self._get_company_data(company_name)
            if not company_data:
                return {'error': f"Company '{company_name}' not found", 'available_companies': self._get_available_companies()}

            predictions = self.make_predictions(company_data)

            report = {
                'company_name': company_name,
                'industry': company_data.get('industry', 'Unknown'),
                'year': company_data.get('year', 'Unknown'),
                'ai_adoption_percentage': company_data.get('ai_adoption_percentage', 'Unknown'),
                'primary_ai_application': company_data.get('primary_ai_application', 'Unknown'),
                'esg_score': company_data.get('esg_score', 'Unknown'),
                'primary_esg_impact': company_data.get('primary_esg_impact', 'Unknown'),
                'sustainable_growth_index': company_data.get('sustainable_growth_index', 'Unknown'),
                'innovation_index': company_data.get('innovation_index', 'Unknown'),
                'predictions': predictions,
                'recommendations': self._generate_recommendations(company_data, predictions)
            }

            # Add any additional columns that exist in the data
            for key, value in company_data.items():
                if key not in report:
                    report[key] = value

            logger.info(f"Sustainability report generated for {company_name}")
            return report
        except Exception as e:
            logger.error(f"Error in generating sustainability report: {str(e)}")
            return {'error': str(e), 'available_companies': self._get_available_companies()}

    def _get_company_data(self, company_name):
        if 'gen_ai_business' not in self.data_processor.data or self.data_processor.data['gen_ai_business'] is None:
            return None
        company_data = self.data_processor.data['gen_ai_business'][self.data_processor.data['gen_ai_business']['company'] == company_name]
        return company_data.iloc[0].to_dict() if not company_data.empty else None

    def _generate_recommendations(self, company_data, predictions):
        recommendations = []

        if 'esg_score' in company_data and 'ai_esg_alignment' in predictions:
            if predictions['ai_esg_alignment'] > company_data['esg_score']:
                recommendations.append("Consider increasing AI investments in ESG initiatives to improve overall ESG performance.")

        if 'ai_impact' in predictions and predictions['ai_impact'] > 50:
            recommendations.append("Explore opportunities to leverage AI for process efficiency improvements across the organization.")

        if 'gen_ai_business' in predictions and predictions['gen_ai_business'] > 0.5:
            recommendations.append("Focus on integrating generative AI into your business model to drive sustainable growth.")

        if 'ai_adoption_percentage' in company_data and company_data['ai_adoption_percentage'] < 50:
            recommendations.append("Increase AI adoption across the organization to stay competitive and drive innovation.")

        return recommendations

    def make_predictions(self, input_data):
        predictions = {}
        for key, model in self.models.items():
            preprocessor = self.data_processor.preprocessors[key]
            features = self.data_processor.feature_names[key]
            
            # Ensure we only use features that are present in the input data
            available_features = [f for f in features if f in input_data]
            
            if not available_features:
                self.logger.warning(f"No matching features found for {key} model. Skipping prediction.")
                continue

            input_df = pd.DataFrame([{f: input_data[f] for f in available_features}])
            input_preprocessed = preprocessor.transform(input_df)
            
            prediction = model.predict(input_preprocessed)[0]
            predictions[key] = prediction
        
        return predictions

    

    def _get_available_companies(self):
        if 'gen_ai_business' not in self.data_processor.data or self.data_processor.data['gen_ai_business'] is None:
            return []
        return self.data_processor.data['gen_ai_business']['company'].tolist()

    

    @classmethod
    def run(cls):
        model = cls()
        model.load_data()
        model.preprocess_data()
        model.train_models()
        model.train_time_series_model()
        return model

class TextGenerator:
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.logger = enhanced_model.logger

    def generate(self, prompt: str, max_length: int = 100) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
            outputs = self.model(**inputs)
            
            generated_text = self.tokenizer.decode(outputs.last_hidden_state[0].argmax(dim=-1))
            
            return generated_text
        except Exception as e:
            self.logger.error(f"Error in text generation: {str(e)}")
            return f"Error in text generation: {str(e)}"

class PredictiveAnalytics:
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.logger = enhanced_model.logger

    def predict(self, data: pd.DataFrame, dataset_key: str) -> dict:
        try:
            model = self.enhanced_model.models.get(dataset_key)
            preprocessor = self.enhanced_model.data_processor.preprocessors.get(dataset_key)
            
            if model is None or preprocessor is None:
                raise ValueError(f"No trained model or preprocessor found for {dataset_key}")
            
            X_processed = preprocessor.transform(data)
            predictions = model.predict(X_processed)
            
            feature_importance = self._get_feature_importance(model, dataset_key)
            
            return {
                'predictions': predictions.tolist(),
                'feature_importance': feature_importance,
            }
        except Exception as e:
            self.logger.error(f"Error in making predictions: {str(e)}")
            return {'error': str(e)}

    def _get_feature_importance(self, model, dataset_key):
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.enhanced_model.data_processor.feature_names[dataset_key], model.feature_importances_))
        return {}

class EnvironmentalImpactAnalyzer:
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.logger = enhanced_model.logger

    def analyze(self, company: str, year: int) -> Dict[str, Any]:
        try:
            company_data = self._get_company_data(company, year)
            if company_data is None:
                return {'error': f"No data found for company {company} in year {year}"}

            impact_score = self._calculate_impact_score(company_data)
            recommendations = self._generate_recommendations(impact_score, company_data)
            
            return {
                'company': company,
                'year': year,
                'impact_score': impact_score,
                'ai_adoption_percentage': company_data['ai_adoption_percentage'],
                'primary_ai_application': company_data['primary_ai_application'],
                'esg_score': company_data['esg_score'],
                'primary_esg_impact': company_data['primary_esg_impact'],
                'recommendations': recommendations
            }
        except Exception as e:
            self.logger.error(f"Error in environmental impact analysis: {str(e)}")
            return {'error': str(e)}

    def _get_company_data(self, company: str, year: int) -> Dict[str, Any]:
        data = self.enhanced_model.data_processor.data['gen_ai_business']
        company_data = data[(data['company'] == company) & (data['year'] == year)]
        return company_data.iloc[0].to_dict() if not company_data.empty else None

    def _calculate_impact_score(self, company_data: Dict[str, Any]) -> float:
        return (company_data['esg_score'] * 0.4 + 
                company_data['ai_adoption_percentage'] * 0.3 + 
                company_data['sustainable_growth_index'] * 0.3)

    def _generate_recommendations(self, impact_score: float, company_data: Dict[str, Any]) -> List[str]:
        recommendations = []
        if impact_score < 50:
            recommendations.append(f"Increase focus on {company_data['primary_esg_impact']} to improve overall environmental impact.")
        if company_data['ai_adoption_percentage'] < 50:
            recommendations.append(f"Consider expanding use of AI in {company_data['primary_ai_application']} to drive efficiency and sustainability.")
        if company_data['sustainable_growth_index'] < 0.5:
            recommendations.append("Develop strategies to improve sustainable growth, possibly by leveraging AI technologies.")
        return recommendations

class InnovativeBusinessModelGenerator:
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.logger = enhanced_model.logger
        self.genai_model = genai.GenerativeModel('gemini-pro')

    def generate_business_model(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._validate_input(input_data)
            company_data = self._get_company_data(input_data['company'], input_data['year'])
            if company_data is None:
                return {'error': f"No data found for company {input_data['company']} in year {input_data['year']}"}

            market_insights = self._analyze_market(company_data)
            sustainability_metrics = self._calculate_sustainability_metrics(company_data)
            innovation_score = self._calculate_innovation_score(company_data)
            future_trends = self._forecast_future_trends(company_data)
            
            prompt = f"""
            Generate an innovative, AI-driven business model for {input_data['company']} in the {company_data['industry']} industry.
            Consider the following:
            1. Current AI adoption rate: {company_data['ai_adoption_percentage']}%
            2. Primary AI application: {company_data['primary_ai_application']}
            3. ESG score: {sustainability_metrics['esg_score']}
            4. Sustainable growth index: {sustainability_metrics['sustainable_growth_index']}
            5. Innovation score: {innovation_score}
            6. Industry average AI adoption: {market_insights['industry_average_ai_adoption']}%

            Focus on:
            - Leveraging generative AI technologies
            - Disrupting traditional industry practices
            - Aligning with ESG objectives
            - Fostering sustainable growth
            - Creating new revenue streams
            - Improving efficiency through AI
            """

            response = self.genai_model.generate_content(prompt)
            business_model = self._parse_generated_model(response.text)
            
            return {
                'business_model': business_model,
                'market_insights': market_insights,
                'sustainability_metrics': sustainability_metrics,
                'innovation_score': innovation_score,
                'future_trends': future_trends
            }
        except Exception as e:
            self.logger.error(f"Error in business model generation: {str(e)}")
            return {'error': str(e)}

    def _parse_generated_model(self, generated_text: str) -> Dict[str, Any]:
        # Implement parsing logic to extract structured information from the generated text
        # This is a placeholder implementation
        return {
            "value_proposition": generated_text[:100],
            "key_activities": generated_text[100:200].split(', '),
            "revenue_streams": generated_text[200:300].split(', '),
            "sustainability_impact": generated_text[300:400],
            "ai_integration": generated_text[400:500],
        }

    def _validate_input(self, data: Dict[str, Any]):
        required_fields = ['company', 'year']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    def _get_company_data(self, company: str, year: int) -> Dict[str, Any]:
        data = self.enhanced_model.data_processor.data['gen_ai_business']
        company_data = data[(data['company'] == company) & (data['year'] == year)]
        return company_data.iloc[0].to_dict() if not company_data.empty else None

    def _analyze_market(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        industry_data = self.enhanced_model.data_processor.data['gen_ai_business'][
            self.enhanced_model.data_processor.data['gen_ai_business']['industry'] == company_data['industry']
        ]
        return {
            'industry_average_growth': industry_data['revenue_growth'].mean(),
            'industry_average_ai_adoption': industry_data['ai_adoption_percentage'].mean(),
            'industry_average_esg_score': industry_data['esg_score'].mean()
        }

    def _calculate_sustainability_metrics(self, company_data: Dict[str, Any]) -> Dict[str, float]:
        return {
            'esg_score': company_data['esg_score'],
            'sustainable_growth_index': company_data['sustainable_growth_index']
        }

    def _calculate_innovation_score(self, company_data: Dict[str, Any]) -> float:
        return (company_data['ai_adoption_percentage'] * 0.4 + 
                company_data['innovation_index'] * 0.6)

    def _forecast_future_trends(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.enhanced_model.prophet_model:
            future_dates = self.enhanced_model.prophet_model.make_future_dataframe(periods=5, freq='Y')
            forecast = self.enhanced_model.prophet_model.predict(future_dates)
            return {
                'sustainable_growth_forecast': forecast['yhat'].tolist()[-5:],
                'forecast_dates': forecast['ds'].dt.year.tolist()[-5:]
            }
        return {}

import base64
import os
from datetime import datetime

class GenerativeImageCreator:
    def __init__(self):
        self.stability_api = stability_api
        self.image_save_path = "generated_images"  # Directory to save images
        os.makedirs(self.image_save_path, exist_ok=True)  # Create the folder if it doesn't exist

    def create_image(self, prompt: str) -> str:
        try:
            response = self.stability_api.generate(
                prompt=prompt,
                steps=30,
                cfg_scale=8.0,
                width=512,
                height=512,
                samples=1,
                sampler=generation.SAMPLER_K_DPMPP_2M
            )

            for resp in response:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        return "Image generation failed due to content filter"
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        # Generate a unique filename using the current timestamp
                        filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                        file_path = os.path.join(self.image_save_path, filename)

                        # Save the binary image data to a file
                        with open(file_path, 'wb') as image_file:
                            image_file.write(artifact.binary)

                        # Return the file path or URL where the image is saved
                        return f"Image saved at {file_path}"

            return "Image generation failed"
        except Exception as e:
            logger.error(f"Error in image generation: {str(e)}")
            return f"Error in image generation: {str(e)}"


# Remove the main execution block and replace it with:
def initialize_model():
    return EnhancedSustainabilityModel.run()

# This function will be called when the server starts
initialized_model = initialize_model()