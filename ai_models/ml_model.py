import os
import gc
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel, f_classif
from sklearn.decomposition import PCA
import lightgbm as lgb
import shap
from joblib import dump, load
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from dotenv import load_dotenv
import google.generativeai as genai
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification
import torch
from typing import Dict, List, Any, Union
from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd
import re
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.stats import pearsonr
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import shap
import geopandas as gpd

warnings.filterwarnings('ignore')

# Load environment variables and configure APIs
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
stability_api = client.StabilityInference(key=os.getenv('STABILITY_KEY'), verbose=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    @staticmethod
    def load_data(csv_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded data from {csv_path}. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {csv_path}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def find_csv_file(filename):
        current_dir = os.getcwd()
        for root, dirs, files in os.walk(current_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

class TextGenerator:
    def __init__(self):
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def generate(self, prompt: str, csv_file: str = None, use_gpt2: bool = False) -> str:
        try:
            if csv_file:
                csv_path = DataLoader.find_csv_file(f"{csv_file}_dataset.csv")
                if csv_path:
                    df = DataLoader.load_data(csv_path)
                    context = f"Based on the following data:\n{df.head().to_string()}\n\n"
                    prompt = context + prompt
                else:
                    raise ValueError(f"Invalid CSV file name: {csv_file}")
            
            if use_gpt2:
                return self._generate_with_gpt2(prompt)
            else:
                response = self.gemini_model.generate_content(prompt)
                return response.text
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return f"Error in text generation: {str(e)}"

    def _generate_with_gpt2(self, prompt: str) -> str:
        try:
            input_ids = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
            
            output = self.gpt2_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            
            return self.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error in GPT-2 text generation: {str(e)}")
            return f"Error in GPT-2 text generation: {str(e)}"

class ImageGenerator:
    def generate(self, prompt: str) -> str:
        try:
            response = stability_api.generate(prompt=prompt)
            for resp in response:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        raise ValueError("Your request activated the API's safety filters and could not be processed.")
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img_data = artifact.binary
                        os.makedirs('generated_images', exist_ok=True)
                        img_path = f"generated_images/{prompt[:20]}.png"
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        return f"http://example.com/{img_path}"
        except Exception as e:
            logger.error(f"Error in image generation: {str(e)}")
            return f"Error in image generation: {str(e)}"

class PredictiveAnalytics:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(random_state=42),
            'lgbm': lgb.LGBMRegressor(random_state=42)
        }
        self.best_model = None
        self.preprocessor = None
        self.feature_names = None

    def train(self, data: pd.DataFrame, target_column: str) -> dict:
        try:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            self.preprocessor = ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])
            
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            
            self.feature_names = (numeric_features.tolist() +
                                  self.preprocessor.named_transformers_['cat']
                                  .named_steps['onehot']
                                  .get_feature_names(categorical_features).tolist())
            
            results = {}
            for name, model in self.models.items():
                model.fit(X_train_processed, y_train)
                train_score = model.score(X_train_processed, y_train)
                test_score = model.score(X_test_processed, y_test)
                y_pred = model.predict(X_test_processed)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'mse': mse,
                    'r2': r2
                }
            
            best_model_name = max(results, key=lambda x: results[x]['test_score'])
            self.best_model = self.models[best_model_name]
            
            logger.info(f"Best model: {best_model_name}")
            
            return results
        except Exception as e:
            logger.error(f"Error in training predictive model: {str(e)}")
            return {'error': str(e)}

    def predict(self, data: pd.DataFrame) -> dict:
        try:
            if self.best_model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")
            
            X_processed = self.preprocessor.transform(data)
            predictions = self.best_model.predict(X_processed)
            feature_importance = self._get_feature_importance()
            shap_values = self._get_shap_values(X_processed)
            
            return {
                'predictions': predictions.tolist(),
                'feature_importance': feature_importance,
                'shap_values': shap_values
            }
        except Exception as e:
            logger.error(f"Error in making predictions: {str(e)}")
            return {'error': str(e)}

    def _get_feature_importance(self) -> dict:
        if hasattr(self.best_model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.best_model.feature_importances_))
        return {}

    def _get_shap_values(self, X_processed) -> dict:
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X_processed)
        return {
            'shap_values': shap_values,
            'feature_names': self.feature_names
        }

class EnvironmentalImpactAnalyzer:
    def __init__(self):
        self.air_quality_data = DataLoader.load_data(DataLoader.find_csv_file('air_quality_dataset.csv'))
        self.energy_data = DataLoader.load_data(DataLoader.find_csv_file('energy_dataset.csv'))
        self.climate_change_data = DataLoader.load_data(DataLoader.find_csv_file('world_bank_climate_change_dataset.csv'))

    def analyze(self, country: str, year: int) -> Dict[str, Any]:
        try:
            air_quality = self._get_air_quality(country)
            energy_metrics = self._get_energy_metrics(country, year)
            climate_indicators = self._get_climate_indicators(country, year)
            
            impact_score = self._calculate_impact_score(air_quality, energy_metrics, climate_indicators)
            recommendations = self._generate_recommendations(impact_score, air_quality, energy_metrics, climate_indicators)
            
            return {
                'impact_score': impact_score,
                'air_quality': air_quality,
                'energy_metrics': energy_metrics,
                'climate_indicators': climate_indicators,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Error in environmental impact analysis: {str(e)}")
            return {'error': str(e)}

    def _get_air_quality(self, country: str) -> Dict[str, float]:
        country_data = self.air_quality_data[self.air_quality_data['country_name'] == country]
        if country_data.empty:
            return {'aqi_value': None, 'co_aqi_value': None, 'no2_aqi_value': None, 'pm2.5_aqi_value': None}
        return {
            'aqi_value': country_data['aqi_value'].mean(),
            'co_aqi_value': country_data['co_aqi_value'].mean(),
            'no2_aqi_value': country_data['no2_aqi_value'].mean(),
            'pm2.5_aqi_value': country_data['pm2.5_aqi_value'].mean()
        }

    def _get_energy_metrics(self, country: str, year: int) -> Dict[str, float]:
        country_data = self.energy_data[(self.energy_data['country'] == country) & (self.energy_data['year'] == year)]
        if country_data.empty:
            return {'energy_per_capita': None, 'renewable_electricity': None}
        return {
            'energy_per_capita': country_data['energy_per_capita'].values[0],
            'renewable_electricity': country_data['renewable_electricity'].values[0]
        }

    def _get_climate_indicators(self, country: str, year: int) -> Dict[str, float]:
        country_data = self.climate_change_data[(self.climate_change_data['Country Name'] == country) & (self.climate_change_data['year'] == year)]
        if country_data.empty:
            return {'urban_population_pct': None}
        return {
            'urban_population_pct': country_data['Urban population (% of total population)'].values[0]
        }

    def _calculate_impact_score(self, air_quality: Dict[str, float], energy_metrics: Dict[str, float], climate_indicators: Dict[str, float]) -> float:
        # Simplified scoring method - can be made more sophisticated
        score = 0
        if air_quality['aqi_value']:
            score -= air_quality['aqi_value'] / 50  # Lower AQI is better
        if energy_metrics['renewable_electricity']:
            score += energy_metrics['renewable_electricity'] / 20  # Higher renewable % is better
        if climate_indicators['urban_population_pct']:
            score -= climate_indicators['urban_population_pct'] / 200  # Assume higher urbanization has slightly negative impact
        return max(min(score, 10), 0)  # Normalize score between 0 and 10

    def _generate_recommendations(self, impact_score: float, air_quality: Dict[str, float], energy_metrics: Dict[str, float], climate_indicators: Dict[str, float]) -> List[str]:
        recommendations = []
        if impact_score < 5:
            if air_quality['aqi_value'] and air_quality['aqi_value'] > 50:
                recommendations.append("Implement stricter air quality control measures")
            if energy_metrics['renewable_electricity'] and energy_metrics['renewable_electricity'] < 20:
                recommendations.append("Increase investment in renewable energy sources")
            if climate_indicators['urban_population_pct'] and climate_indicators['urban_population_pct'] > 80:
                recommendations.append("Develop sustainable urban planning strategies")
        else:
            recommendations.append("Maintain current environmental policies and continue to improve")
        return recommendations
class ESGScoreCalculator:
    def __init__(self):
        self.esg_data = DataLoader.load_data(DataLoader.find_csv_file('esg_score_dataset.csv'))
        self.companies_data = DataLoader.load_data(DataLoader.find_csv_file('companies_dataset.csv'))
        self.air_quality_data = DataLoader.load_data(DataLoader.find_csv_file('air_quality_dataset.csv'))
        self.energy_data = DataLoader.load_data(DataLoader.find_csv_file('energy_dataset.csv'))
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.preprocessor = None

    def preprocess_data(self):
        # Merge datasets
        merged_data = self.esg_data.merge(self.companies_data, left_on='Name', right_on='name', how='left')
        merged_data = merged_data.merge(self.air_quality_data, left_on='country', right_on='country_name', how='left')
        merged_data = merged_data.merge(self.energy_data, on='country', how='left')

        # Select features
        features = ['Full Time Employees', 'Controversy Level', 'aqi_value', 'energy_per_capita', 'renewable_electricity', 'gdp']
        target = 'Total ESG Risk score'

        X = merged_data[features]
        y = merged_data[target]

        # Handle missing values and encode categorical variables
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        self.preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

        X_processed = self.preprocessor.fit_transform(X)
        return X_processed, y

    def train_model(self):
        X_processed, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Model performance - MSE: {mse}, R2: {r2}")

    def calculate_esg_score(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            input_data = pd.DataFrame([company_data])
            processed_input = self.preprocessor.transform(input_data)
            esg_score = self.model.predict(processed_input)[0]

            # Calculate component scores
            feature_importance = self.model.feature_importances_
            environmental_score = np.dot(processed_input[0, :3], feature_importance[:3]) / np.sum(feature_importance[:3])
            social_score = np.dot(processed_input[0, 3:6], feature_importance[3:6]) / np.sum(feature_importance[3:6])
            governance_score = np.dot(processed_input[0, 6:], feature_importance[6:]) / np.sum(feature_importance[6:])

            return {
                'total_esg_score': esg_score,
                'environmental_score': environmental_score,
                'social_score': social_score,
                'governance_score': governance_score
            }
        except Exception as e:
            logging.error(f"Error in ESG score calculation: {str(e)}")
            return {'error': str(e)}

class InnovativeBusinessModelGenerator:
    def __init__(self):
        self.esg_calculator = ESGScoreCalculator()
        self.startups_data = DataLoader.load_data(DataLoader.find_csv_file('innovative_startups_dataset.csv'))
        self.sdg_data = DataLoader.load_data(DataLoader.find_csv_file('sdg_indicator_dataset.csv'))
        self.world_bank_data = DataLoader.load_data(DataLoader.find_csv_file('world_bank_climate_change_dataset.csv'))
        self.nlp_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.nlp_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def generate_business_model(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._validate_input(input_data)
            integrated_data = self._integrate_data(input_data)
            market_insights = self._analyze_market(integrated_data)
            sustainability_metrics = self._calculate_sustainability_metrics(integrated_data)
            innovation_score = self._calculate_innovation_score(integrated_data)
            future_trends = self._forecast_future_trends(integrated_data)
            
            business_model = self._generate_business_model_content(input_data, market_insights, sustainability_metrics, innovation_score, future_trends)
            
            return {
                'business_model': business_model,
                'market_insights': market_insights,
                'sustainability_metrics': sustainability_metrics,
                'innovation_score': innovation_score,
                'future_trends': future_trends
            }
        except Exception as e:
            logging.error(f"Error in business model generation: {str(e)}")
            return self._generate_fallback_insights(str(e))

    def _validate_input(self, data: Dict[str, Any]):
        required_fields = ['industry', 'target_market', 'key_resources']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    def _integrate_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        industry_startups = self.startups_data[self.startups_data['Industry'] == input_data['industry']]
        relevant_sdg = self.sdg_data[self.sdg_data['Indicator Code'] == '17.16.1']
        country_data = self.world_bank_data[self.world_bank_data['Country Name'] == input_data.get('country', 'World')]

        integrated_data = pd.concat([industry_startups, relevant_sdg, country_data], axis=1)
        integrated_data['input_description'] = input_data.get('description', '')
        return integrated_data

    def _analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        market_size = data['Last Valuation (Billion $)'].sum()
        top_companies = data.nlargest(5, 'Last Valuation (Billion $)')['Company'].tolist()
        avg_valuation = data['Last Valuation (Billion $)'].mean()
        
        return {
            'market_size': market_size,
            'top_companies': top_companies,
            'average_valuation': avg_valuation
        }

    def _calculate_sustainability_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        company_data = {
            'Full Time Employees': data['Full Time Employees'].mean(),
            'Controversy Level': data['Controversy Level'].mode()[0] if 'Controversy Level' in data else 0,
            'aqi_value': data['aqi_value'].mean() if 'aqi_value' in data else 0,
            'energy_per_capita': data['energy_per_capita'].mean() if 'energy_per_capita' in data else 0,
            'renewable_electricity': data['renewable_electricity'].mean() if 'renewable_electricity' in data else 0,
            'gdp': data['gdp'].mean() if 'gdp' in data else 0
        }
        return self.esg_calculator.calculate_esg_score(company_data)

    def _calculate_innovation_score(self, data: pd.DataFrame) -> float:
        if 'Year Joined' in data.columns and 'Last Valuation (Billion $)' in data.columns:
            age = datetime.now().year - data['Year Joined'].mean()
            valuation = data['Last Valuation (Billion $)'].mean()
            return (valuation / age) * 100 if age > 0 else 0
        return 0

    def _forecast_future_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        if 'Urban population (% of total population)' in data.columns:
            urban_pop_data = data[['year', 'Urban population (% of total population)']].dropna()
            urban_pop_data = urban_pop_data.rename(columns={'year': 'ds', 'Urban population (% of total population)': 'y'})
            model = Prophet()
            model.fit(urban_pop_data)
            future_dates = model.make_future_dataframe(periods=5, freq='Y')
            forecast = model.predict(future_dates)
            return {'urban_population_forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_dict('records')}
        return {}

    def _generate_business_model_content(self, input_data: Dict[str, Any], market_insights: Dict[str, Any], 
                                         sustainability_metrics: Dict[str, float], innovation_score: float, 
                                         future_trends: Dict[str, Any]) -> Dict[str, Any]:
        # Generate a structured business model based on the insights
        value_proposition = f"Innovative {input_data['industry']} solution addressing {input_data['target_market']} needs with a focus on sustainability (ESG Score: {sustainability_metrics['total_esg_score']:.2f})"
        
        revenue_streams = [
            f"Subscription-based {input_data['industry']} services",
            "Consulting on sustainable practices",
            "Data analytics products"
        ]
        
        key_activities = [
            f"R&D in {input_data['industry']} technologies",
            "Sustainable product development",
            "Customer engagement and education"
        ]
        
        customer_segments = [
            input_data['target_market'],
            "Environmentally conscious consumers",
            "B2B clients seeking sustainable solutions"
        ]
        
        key_partners = [
            "Sustainability research institutions",
            f"Leading {input_data['industry']} companies: {', '.join(market_insights['top_companies'][:3])}",
            "Environmental NGOs"
        ]
        
        cost_structure = [
            "R&D investments",
            "Sustainable material sourcing",
            "Marketing and customer acquisition"
        ]
        
        channels = [
            "Direct online sales",
            "Partnership networks",
            "Sustainability forums and events"
        ]
        
        sustainability_impact = f"Potential to reduce industry environmental impact by {sustainability_metrics['environmental_score']*10:.1f}%"
        
        competitive_advantage = f"Unique combination of {input_data['industry']} innovation (Innovation Score: {innovation_score:.2f}) and strong sustainability focus"
        
        scalability_plan = f"Expand to new markets leveraging the projected urban population growth of {future_trends.get('urban_population_forecast', [{}])[0].get('yhat', 0):.1f}% in 5 years"
        
        return {
            "value_proposition": value_proposition,
            "revenue_streams": revenue_streams,
            "key_activities": key_activities,
            "customer_segments": customer_segments,
            "key_partners": key_partners,
            "cost_structure": cost_structure,
            "channels": channels,
            "sustainability_impact": sustainability_impact,
            "competitive_advantage": competitive_advantage,
            "scalability_plan": scalability_plan
        }

    def _generate_fallback_insights(self, error_message: str) -> Dict[str, Any]:
        return {
            "error": f"Unable to generate business model due to an error: {error_message}. Using fallback data.",
            "fallback_model": {
                "value_proposition": "AI-driven sustainable business solutions for emerging markets",
                "revenue_streams": ["Subscription-based services", "Consulting fees", "Data analytics products"],
                "key_activities": ["AI model development", "Market research", "Sustainability consulting"],
                "customer_segments": ["Small to medium enterprises", "Tech startups", "Sustainability-focused corporations"],
                "key_partners": ["Local tech incubators", "Environmental NGOs", "University research departments"],
                "cost_structure": ["AI development costs", "Marketing and sales", "Research and development"],
                "channels": ["Direct sales", "Online platform", "Partner networks"],
                "sustainability_impact": "Reducing carbon footprint through AI-optimized resource management",
                "competitive_advantage": "Unique blend of AI technology and sustainability expertise",
                "scalability_plan": "Expand to new emerging markets and develop industry-specific AI solutions"
            }
        }
    
import os
import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
import lightgbm as lgb
from joblib import dump, load

class EnhancedSustainabilityModel:
    def __init__(self):
        self.data = {}
        self.models = {}
        self.preprocessors = {}
        self.feature_names = {}
        self.target_variables = {}
        self.cache_dir = 'data_cache'
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def load_data(self):
        csv_files = {
            'air_quality': 'air_quality_dataset.csv',
            'companies': 'companies_dataset.csv',
            'energy': 'energy_dataset.csv',
            'esg_score': 'esg_score_dataset.csv',
            'innovative_startups': 'innovative_startups_dataset.csv',
            'sdg_indicator': 'sdg_indicator_dataset.csv',
            'world_bank': 'world_bank_climate_change_dataset.csv'
        }
    
        for key, file in csv_files.items():
            cache_file = os.path.join(self.cache_dir, f'{key}_cache.parquet')
            if os.path.exists(cache_file):
                self.data[key] = pd.read_parquet(cache_file)
            else:
                file_path = self.find_csv_file(file)
                if file_path:
                    self.data[key] = pd.read_csv(file_path)
                    self.data[key].to_parquet(cache_file)
                else:
                    self.logger.warning(f"File not found: {file}")
                    self.data[key] = None

        self.logger.info("All datasets loaded")

    def find_csv_file(self, filename):
        for root, _, files in os.walk(os.getcwd()):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def preprocess_data(self):
        for key in self.data:
            if self.data[key] is not None and not self.data[key].empty:
                self.handle_special_cases(key)
                self.handle_missing_values(key)
                self.handle_data_type_inconsistencies(key)
                self.prepare_preprocessor(key)
                self.identify_target_variable(key)

    def handle_special_cases(self, key):
        if key == 'world_bank':
            required_columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
            self.data[key] = pd.melt(self.data[key], id_vars=required_columns, 
                                     var_name='year', value_name='value')
            self.data[key]['year'] = pd.to_numeric(self.data[key]['year'], errors='coerce')
            self.data[key]['value'] = pd.to_numeric(self.data[key]['value'].replace('..', np.nan), errors='coerce')

    def handle_missing_values(self, key):
        if key == 'companies':
            self.data[key]['year founded'] = self.data[key]['year founded'].fillna(self.data[key]['year founded'].median())
        elif key == 'innovative_startups':
            self.data[key]['Company Website'] = self.data[key]['Company Website'].fillna('N/A')
        elif key == 'sdg_indicator':
            numeric_columns = self.data[key].select_dtypes(include=[np.number]).columns
            self.data[key][numeric_columns] = self.data[key][numeric_columns].fillna(self.data[key][numeric_columns].mean())

    def handle_data_type_inconsistencies(self, key):
        if key == 'companies':
            self.data[key]['year founded'] = pd.to_numeric(self.data[key]['year founded'], errors='coerce').astype('Int64')
        elif key == 'world_bank':
            year_columns = [col for col in self.data[key].columns[4:] if col.isdigit()]
            for col in year_columns:
                self.data[key][col] = pd.to_numeric(self.data[key][col], errors='coerce')
        elif key == 'innovative_startups':
            if 'Year Joined' in self.data[key].columns:
                self.data[key]['Year Joined'] = pd.to_numeric(self.data[key]['Year Joined'], errors='coerce').astype('Int64')
            if 'Last Valuation (Billion $)' in self.data[key].columns:
                self.data[key]['Last Valuation (Billion $)'] = pd.to_numeric(self.data[key]['Last Valuation (Billion $)'], errors='coerce')
        elif key == 'sdg_indicator':
            for col in self.data[key].columns[2:]:
                self.data[key][col] = pd.to_numeric(self.data[key][col], errors='coerce')
        else:
            for column in self.data[key].columns:
                if self.data[key][column].dtype == 'object':
                    try:
                        self.data[key][column] = pd.to_numeric(self.data[key][column], errors='coerce')
                    except Exception:
                        pass

    def prepare_preprocessor(self, key):
        numeric_features = self.data[key].select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.data[key].select_dtypes(include=['object']).columns

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

    def identify_target_variable(self, key):
        target_mapping = {
            'air_quality': 'aqi_value',
            'energy': 'energy_per_capita',
            'esg_score': 'Total ESG Risk score',
            'world_bank': 'value',
            'innovative_startups': 'Last Valuation (Billion $)',
            'sdg_indicator': 'sdg_index_score',
            'companies': 'current employee estimate'
        }
        self.target_variables[key] = target_mapping.get(key)
        if self.target_variables[key] is None:
            self.logger.warning(f"No suitable target variable identified for {key} dataset")

    def engineer_features(self):
        # (Feature engineering code remains the same)
        pass

    def explain_model(self, key, model, X):
        feature_names = self.preprocessors[key].get_feature_names_out()
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_)
        else:
            return

        min_length = min(len(feature_importance), len(feature_names))
        importance_df = pd.DataFrame({
            'feature': feature_names[:min_length],
            'importance': feature_importance[:min_length]
        }).sort_values('importance', ascending=False)
        
        self.logger.info(f"Top 5 important features for {key}:")
        self.logger.info(importance_df.head().to_string(index=False))

    def train_models(self):
        for key in self.data:
            if self.data[key] is None or self.data[key].empty or self.target_variables[key] is None:
                continue

            y = self.data[key][self.target_variables[key]]
            mask = ~y.isna()
            X = self.data[key][mask]
            y = y[mask]

            if len(y) == 0:
                continue

            X = self.preprocessors[key].fit_transform(X)
            is_classification = y.dtype == 'object' or len(np.unique(y)) < 10

            if is_classification:
                le = LabelEncoder()
                y = le.fit_transform(y)

            selector = SelectKBest(f_classif if is_classification else f_regression, k=min(50, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            pca = PCA(n_components=min(30, X_selected.shape[1]))
            X_reduced = pca.fit_transform(X_selected)

            X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

            if is_classification:
                models = {
                    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
                    'LogisticRegression': OneVsRestClassifier(LogisticRegression(random_state=42))
                }
                scoring = 'accuracy'
            else:
                models = {
                    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
                    'Lasso': LassoCV(random_state=42),
                    'Ridge': RidgeCV()
                }
                scoring = 'r2'

            best_model = None
            best_score = float('-inf')

            for model_name, model in models.items():
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                mean_cv_score = np.mean(cv_scores)
                
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_model = model

            best_model.fit(X_train, y_train)
            
            y_pred = best_model.predict(X_test)
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                self.logger.info(f"{key} - Test set accuracy: {accuracy:.4f}")
            else:
                r2 = r2_score(y_test, y_pred)
                self.logger.info(f"{key} - Test set R2: {r2:.4f}")

            self.models[key] = best_model
            self.logger.info(f"Best model for {key} dataset: {type(best_model).__name__}")

            self.explain_model(key, best_model, X_test)
            self.save_model(key, best_model)

    def save_model(self, key, model):
        model_filename = f"{key}_model.joblib"
        dump(model, model_filename)

    def load_model(self, key):
        model_filename = f"{key}_model.joblib"
        if os.path.exists(model_filename):
            return load(model_filename)
        else:
            self.logger.warning(f"No saved model found for {key}")
            return None

    def run(self):
        try:
            self.load_data()
            self.preprocess_data()
            self.engineer_features()
            self.train_models()
            self.logger.info("All processes completed successfully")
        except Exception as e:
            self.logger.error(f"An error occurred during execution: {str(e)}")

if __name__ == "__main__":
    model = EnhancedSustainabilityModel()
    model.run()