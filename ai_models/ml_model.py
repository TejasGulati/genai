import os
import logging
from datetime import datetime
import random
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from prophet import Prophet
from joblib import dump, load
from dotenv import load_dotenv
import google.generativeai as genai
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
stability_api = client.StabilityInference(key=os.getenv('STABILITY_KEY'), verbose=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSustainabilityModel:
    def __init__(self):
        self.data = {}
        self.models = {}
        self.preprocessors = {}
        self.feature_names = {}
        self.target_variables = {}
        self.cache_dir = 'data_cache'
        self.prophet_model = None
        
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
            
            # Force reload of World Bank data
            if key in ['world_bank', 'companies']:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    self.logger.info(f"Removed existing cache file for {key} dataset")

            if os.path.exists(cache_file):
                self.data[key] = pd.read_parquet(cache_file)
                self.logger.info(f"Loaded {key} dataset from cache. Shape: {self.data[key].shape}")
            else:
                file_path = self.find_csv_file(file)
                if file_path:
                    self.logger.info(f"Loading {key} dataset from file: {file_path}")
                    if key == 'world_bank':
                        df = pd.read_csv(file_path)
                        self.logger.info(f"World Bank data initial shape: {df.shape}")
                        
                        # Identify year columns
                        year_columns = [col for col in df.columns if col.isdigit()]
                        
                        # Melt the DataFrame to convert years to a single column
                        id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
                        df = pd.melt(df, id_vars=id_vars, value_vars=year_columns, var_name='year', value_name='value')
                        
                        # Convert year and value to numeric and drop NaNs
                        df['year'] = pd.to_numeric(df['year'], errors='coerce')
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        df = df.dropna(subset=['year', 'value'])
                        
                        # Sample approximately 20,000 rows
                        if len(df) > 20000:
                            df = df.sample(n=20000, random_state=42)
                        
                        self.data[key] = df
                    elif key == 'companies':
                        df = pd.read_csv(file_path)
                        # Sample approximately 20,000 rows
                        if len(df) > 20000:
                            df = df.sample(n=20000, random_state=42)
                        self.data[key] = df
                    else:
                        self.data[key] = pd.read_csv(file_path)
                    
                    if self.data[key] is not None and not self.data[key].empty:
                        self.data[key].to_parquet(cache_file)
                        self.logger.info(f"Saved {key} dataset to cache. Shape: {self.data[key].shape}")
                    else:
                        self.logger.warning(f"Dataset {key} is empty or None. Not saving to cache.")
                else:
                    self.logger.warning(f"File not found: {file}")
                    self.data[key] = None

        self.logger.info("All datasets loaded")
        for key, df in self.data.items():
            if df is not None:
                self.logger.info(f"{key} dataset shape: {df.shape}")
            else:
                self.logger.warning(f"{key} dataset is None")

    def find_csv_file(self, filename):
        for root, dirs, files in os.walk('.'):
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
            # Convert 'year' column to numeric
            self.data[key]['year'] = pd.to_numeric(self.data[key]['year'], errors='coerce')
            
            # Convert 'value' column to numeric
            self.data[key]['value'] = pd.to_numeric(self.data[key]['value'], errors='coerce')
            
            # Drop rows where year or value is NaN
            self.data[key] = self.data[key].dropna(subset=['year', 'value'])
            
            self.logger.info(f"Preprocessed World Bank data. Shape: {self.data[key].shape}")

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

    def run_geospatial_analysis(self):
        try:
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(
                self.data['air_quality'], 
                geometry=gpd.points_from_xy(self.data['air_quality'].longitude, self.data['air_quality'].latitude)
            )

            # Perform spatial join with country boundaries
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            gdf = gpd.sjoin(gdf, world, how="inner", op='within')

            # Calculate average AQI by country
            country_aqi = gdf.groupby('country_name')['aqi_value'].mean().sort_values(ascending=False)

            self.logger.info("Geospatial analysis completed")
            return country_aqi.to_dict()
        except Exception as e:
            self.logger.error(f"Error in geospatial analysis: {str(e)}")
            return {'error': str(e)}

    def train_time_series_model(self):
        try:
            # Get the world bank data
            df = self.data['world_bank']
            self.logger.info(f"World Bank data shape: {df.shape}")
            self.logger.info(f"World Bank data columns: {df.columns.tolist()}")

            # Select relevant indicators for sustainability and GenAI
            relevant_indicators = [
                'Urban population (% of total population)',
                'Urban population growth (annual %)',
                'Population growth (annual %)',
                'CO2 emissions (metric tons per capita)',
                'CO2 emissions from liquid fuel consumption (% of total)',
                'CO2 emissions from liquid fuel consumption (kt)',
                'Renewable energy consumption (% of total final energy consumption)',
                'Renewable electricity output (% of total electricity output)',
                'Access to electricity (% of population)',
                'Foreign direct investment, net inflows (% of GDP)',
                'Forest area (% of land area)',
                'Land area where elevation is below 5 meters (% of total land area)',
                'Urban land area where elevation is below 5 meters (% of total land area)',
                'Agricultural land (% of land area)',
                'Internet users (per 100 people)',
                'Research and development expenditure (% of GDP)'
            ]

            # Prepare data for Prophet
            prophet_data = []
            for indicator in relevant_indicators:
                df_filtered = df[df['Indicator Name'] == indicator]
                if not df_filtered.empty:
                    df_prophet = df_filtered[['year', 'value', 'Country Name']].rename(columns={'year': 'ds', 'value': 'y', 'Country Name': 'country'})
                    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
                    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
                    df_prophet['indicator'] = indicator
                    prophet_data.append(df_prophet)

            df_prophet_all = pd.concat(prophet_data)
            df_prophet_all = df_prophet_all.dropna().sort_values(['country', 'indicator', 'ds'])

            # Check if we have sufficient data points
            if len(df_prophet_all) < 100:  # Arbitrary threshold, adjust as needed
                raise ValueError(f"Insufficient data points for time series modeling. Only {len(df_prophet_all)} valid data points found.")

            # Log data preparation results
            self.logger.info(f"Prepared data for time series modeling:")
            self.logger.info(f"Number of indicators with data: {len(set(df_prophet_all['indicator']))}")
            self.logger.info(f"Number of data points: {len(df_prophet_all)}")
            self.logger.info(f"Date range: {df_prophet_all['ds'].min()} to {df_prophet_all['ds'].max()}")

            def get_adaptive_min_points(indicator):
                # Example logic for adaptive minimum points
                if 'CO2 emissions' in indicator:
                    return 3  # Require fewer points for CO2 data
                elif 'Urban population' in indicator:
                    return 4  # Require more points for urban population data
                else:
                    return 5  # Default minimum

            # Train Prophet models for each country and indicator
            self.prophet_models = {}
            successful_trainings = []
            insufficient_data_count = 0
            self.model_confidence = {}

            for country in df_prophet_all['country'].unique():
                self.prophet_models[country] = {}
                for indicator in df_prophet_all[df_prophet_all['country'] == country]['indicator'].unique():
                    df_country_indicator = df_prophet_all[(df_prophet_all['country'] == country) & (df_prophet_all['indicator'] == indicator)]
                    min_data_points = get_adaptive_min_points(indicator)
                    
                    if len(df_country_indicator) >= min_data_points:
                        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
                        model.fit(df_country_indicator[['ds', 'y']])
                        self.prophet_models[country][indicator] = model
                        successful_trainings.append(f"{country} - {indicator}")
                        
                        # Calculate a simple confidence metric based on data points
                        confidence = min(1.0, len(df_country_indicator) / 10)  # Max confidence at 10+ points
                        self.model_confidence[(country, indicator)] = confidence
                    else:
                        insufficient_data_count += 1

            # Log successful trainings and insufficient data information
            self.logger.info(f"Successfully trained {len(successful_trainings)} Prophet models:")
            for training in successful_trainings[:10]:
                country, indicator = training.split(" - ")
                confidence = self.model_confidence.get((country, indicator), 0)
                self.logger.info(f"- {training} (Confidence: {confidence:.2f})")
            if len(successful_trainings) > 10:
                self.logger.info(f"... and {len(successful_trainings) - 10} more")
            
            self.logger.info(f"Number of country-indicator combinations with insufficient data: {insufficient_data_count}")

            # Log data distribution information
            data_distribution = df_prophet_all.groupby(['country', 'indicator']).size().describe()
            self.logger.info(f"Data distribution across country-indicator combinations:")
            self.logger.info(f"{data_distribution}")

            # Generate and log some basic forecast metrics for a sample of models
            sample_size = min(5, len(successful_trainings))
            sampled_trainings = random.sample(successful_trainings, sample_size)
            for training in sampled_trainings:
                country, indicator = training.split(" - ")
                model = self.prophet_models[country][indicator]
                future = model.make_future_dataframe(periods=10, freq='Y')
                forecast = model.predict(future)
                last_known_value = df_prophet_all[(df_prophet_all['country'] == country) & (df_prophet_all['indicator'] == indicator)]['y'].iloc[-1]
                forecasted_value = forecast['yhat'].iloc[-1]
                confidence = self.model_confidence.get((country, indicator), 0)
                self.logger.info(f"Sample forecast for {country}, {indicator} (Confidence: {confidence:.2f}):")
                self.logger.info(f"  Last known value: {last_known_value:.2f}")
                self.logger.info(f"  Forecasted value for last date: {forecasted_value:.2f}")

        except Exception as e:
            self.logger.error(f"Error in training time series model: {str(e)}")
            self.logger.exception("Full traceback:")
            self.prophet_models = None

    def make_time_series_forecast(self, country, indicator, periods=365):
        try:
            if self.prophet_models is None or country not in self.prophet_models or indicator not in self.prophet_models[country]:
                raise ValueError(f"No trained model found for {country}, {indicator}")

            model = self.prophet_models[country][indicator]
            future_dates = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future_dates)

            confidence = self.model_confidence.get((country, indicator), 0)
            
            result = {
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records'),
                'confidence': confidence
            }

            return result
        except Exception as e:
            self.logger.error(f"Error in making time series forecast: {str(e)}")
            return {'error': str(e)}

    def generate_sustainability_report(self, company_name: str) -> Dict[str, Any]:
        try:
            company_data = self.data['companies'][self.data['companies']['name'] == company_name].iloc[0]
            esg_data = self.data['esg_score'][self.data['esg_score']['Name'] == company_name].iloc[0]
            
            # Get predictions
            input_data = {**company_data.to_dict(), **esg_data.to_dict()}
            predictions = self.make_predictions(input_data)
            
            # Get air quality data for the company's country
            country_air_quality = self.data['air_quality'][self.data['air_quality']['country_name'] == company_data['country']].mean().to_dict()
            
            # Get energy data for the company's country
            country_energy_data = self.data['energy'][self.data['energy']['country'] == company_data['country']].iloc[-1].to_dict()
            
            # Get urban population data
            urban_pop_data = self.data['world_bank'][
                (self.data['world_bank']['Country Name'] == company_data['country']) & 
                (self.data['world_bank']['Indicator Name'] == 'Urban population (% of total population)')
            ].iloc[-1]['value']
            
            report = {
                'company_name': company_name,
                'industry': company_data['industry'],
                'country': company_data['country'],
                'year_founded': company_data['year founded'],
                'size_range': company_data['size range'],
                'esg_scores': {
                    'total': esg_data['Total ESG Risk score'],
                    'environment': esg_data['Environment Risk Score'],
                    'social': esg_data['Social Risk Score'],
                    'governance': esg_data['Governance Risk Score']
                },
                'sustainability_prediction': predictions['model_predictions']['rf'],
                'key_factors': dict(sorted(predictions['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]),
                'recommendations': self._generate_recommendations(predictions, country_air_quality, country_energy_data),
                'country_metrics': {
                    'air_quality': country_air_quality,
                    'energy': {
                        'renewable_electricity': country_energy_data.get('renewable_electricity', 'N/A'),
                        'energy_per_capita': country_energy_data.get('energy_per_capita', 'N/A'),
                        'co2_emissions': country_energy_data.get('co2_emissions', 'N/A')
                    },
                    'urban_population_percentage': urban_pop_data
                }
            }
            
            # Add SDG indicator information if available
            sdg_data = self.data['sdg_indicator']
            if not sdg_data.empty and 'Indicator Code' in sdg_data.columns:
                relevant_sdg = sdg_data[sdg_data['Indicator Code'] == '17.16.1'].iloc[0]
                report['sdg_indicator'] = {
                    'indicator_code': relevant_sdg['Indicator Code'],
                    'indicator_name': relevant_sdg['Indicator Name'],
                    'sdg_index_score': relevant_sdg['sdg_index_score']
                }
            
            self.logger.info(f"Sustainability report generated for {company_name}")
            return report
        except Exception as e:
            self.logger.error(f"Error in generating sustainability report: {str(e)}")
            return {'error': str(e)}

    def _generate_recommendations(self, predictions: Dict[str, Any], air_quality: Dict[str, float], energy_data: Dict[str, float]) -> List[str]:
        recommendations = []
        top_features = sorted(predictions['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
        
        for feature, importance in top_features:
            if 'emissions' in feature.lower():
                recommendations.append(f"Focus on reducing {feature} to improve sustainability score. Consider setting science-based targets for emissions reduction.")
            elif 'renewable' in feature.lower():
                current_renewable = energy_data.get('renewable_electricity', 0)
                recommendations.append(f"Increase investment in {feature} to boost sustainability performance. Current renewable electricity usage is {current_renewable}%.")
            elif 'governance' in feature.lower():
                recommendations.append(f"Strengthen {feature} practices to enhance overall ESG score. Consider implementing more transparent reporting and ethical business practices.")
            elif 'aqi' in feature.lower():
                aqi_value = air_quality.get('aqi_value', 'N/A')
                recommendations.append(f"Implement measures to improve air quality, particularly {feature}. The current AQI value in your area is {aqi_value}.")
            elif 'urban' in feature.lower():
                recommendations.append(f"Consider the impact of urbanization on sustainability and develop strategies to address {feature}. Focus on sustainable urban development and smart city initiatives.")
        
        # Add general recommendations based on energy data
        energy_per_capita = energy_data.get('energy_per_capita', 0)
        if energy_per_capita > 5000:  # Arbitrary threshold, adjust as needed
            recommendations.append(f"Your energy consumption per capita ({energy_per_capita} kWh) is high. Implement energy efficiency measures across operations.")
        
        co2_emissions = energy_data.get('co2_emissions', 0)
        if co2_emissions > 1000000:  # Arbitrary threshold, adjust as needed
            recommendations.append(f"Your CO2 emissions ({co2_emissions} metric tons) are significant. Develop a comprehensive carbon reduction strategy.")
        
        return recommendations
    
    def get_model(self, key):
        return self.models.get(key)
    
    def get_preprocessor(self, key):
        return self.preprocessors.get(key)
    
    def get_feature_names(self, key):
        return self.feature_names.get(key)

class TextGenerator:
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.logger = enhanced_model.logger

    def generate(self, prompt: str, csv_file: str = None, use_gpt2: bool = False) -> str:
        try:
            if csv_file:
                df = self.enhanced_model.get_dataset(csv_file)
                if df is not None:
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
            self.logger.error(f"Error in text generation: {str(e)}")
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
            self.logger.error(f"Error in GPT-2 text generation: {str(e)}")
            return f"Error in GPT-2 text generation: {str(e)}"
class ImageGenerator:
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.logger = enhanced_model.logger

    def generate(self, prompt: str) -> str:
        try:
            # Use sustainability data to enhance the prompt
            sustainability_context = self.enhanced_model.get_sustainability_context()
            enhanced_prompt = f"{prompt} with consideration for {sustainability_context}"
            
            response = stability_api.generate(prompt=enhanced_prompt)
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
            self.logger.error(f"Error in image generation: {str(e)}")
            return f"Error in image generation: {str(e)}"

class PredictiveAnalytics:
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.logger = enhanced_model.logger

    def predict(self, data: pd.DataFrame, dataset_key: str) -> dict:
        try:
            model = self.enhanced_model.get_model(dataset_key)
            preprocessor = self.enhanced_model.get_preprocessor(dataset_key)
            
            if model is None or preprocessor is None:
                raise ValueError(f"No trained model or preprocessor found for {dataset_key}")
            
            X_processed = preprocessor.transform(data)
            predictions = model.predict(X_processed)
            
            feature_importance = self._get_feature_importance(model, dataset_key)
            shap_values = self._get_shap_values(model, X_processed, dataset_key)
            
            return {
                'predictions': predictions.tolist(),
                'feature_importance': feature_importance,
                'shap_values': shap_values
            }
        except Exception as e:
            self.logger.error(f"Error in making predictions: {str(e)}")
            return {'error': str(e)}

    def _get_feature_importance(self, model, dataset_key):
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.enhanced_model.get_feature_names(dataset_key), model.feature_importances_))
        return {}

    def _get_shap_values(self, model, X_processed, dataset_key):
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        return {
            'shap_values': shap_values,
            'feature_names': self.enhanced_model.get_feature_names(dataset_key)
        }

class EnvironmentalImpactAnalyzer:
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.logger = enhanced_model.logger

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
            self.logger.error(f"Error in environmental impact analysis: {str(e)}")
            return {'error': str(e)}

    def _get_air_quality(self, country: str) -> Dict[str, float]:
        air_quality_data = self.enhanced_model.get_dataset('air_quality')
        country_data = air_quality_data[air_quality_data['country_name'] == country]
        if country_data.empty:
            return {'aqi_value': None, 'co_aqi_value': None, 'no2_aqi_value': None, 'pm2.5_aqi_value': None}
        return {
            'aqi_value': country_data['aqi_value'].mean(),
            'co_aqi_value': country_data['co_aqi_value'].mean(),
            'no2_aqi_value': country_data['no2_aqi_value'].mean(),
            'pm2.5_aqi_value': country_data['pm2.5_aqi_value'].mean()
        }

    def _get_energy_metrics(self, country: str, year: int) -> Dict[str, float]:
        energy_data = self.enhanced_model.get_dataset('energy')
        country_data = energy_data[(energy_data['country'] == country) & (energy_data['year'] == year)]
        if country_data.empty:
            return {'energy_per_capita': None, 'renewable_electricity': None}
        return {
            'energy_per_capita': country_data['energy_per_capita'].values[0],
            'renewable_electricity': country_data['renewable_electricity'].values[0]
        }

    def _get_climate_indicators(self, country: str, year: int) -> Dict[str, float]:
        world_bank_data = self.enhanced_model.get_dataset('world_bank')
        country_data = world_bank_data[(world_bank_data['Country Name'] == country) & (world_bank_data['year'] == year)]
        if country_data.empty:
            return {'urban_population_pct': None}
        urban_pop_data = country_data[country_data['Indicator Name'] == 'Urban population (% of total population)']
        return {
            'urban_population_pct': urban_pop_data['value'].values[0] if not urban_pop_data.empty else None
        }

    def _calculate_impact_score(self, air_quality: Dict[str, float], energy_metrics: Dict[str, float], climate_indicators: Dict[str, float]) -> float:
        score = 0
        if air_quality['aqi_value']:
            score -= air_quality['aqi_value'] / 50
        if energy_metrics['renewable_electricity']:
            score += energy_metrics['renewable_electricity'] / 20
        if climate_indicators['urban_population_pct']:
            score -= climate_indicators['urban_population_pct'] / 200
        return max(min(score, 10), 0)

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
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.logger = enhanced_model.logger

    def calculate_esg_score(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            input_data = pd.DataFrame([company_data])
            esg_model = self.enhanced_model.get_model('esg_score')
            esg_preprocessor = self.enhanced_model.get_preprocessor('esg_score')
            
            if esg_model is None or esg_preprocessor is None:
                raise ValueError("ESG model or preprocessor not found")
            
            processed_input = esg_preprocessor.transform(input_data)
            esg_score = esg_model.predict(processed_input)[0]

            feature_importance = self._get_feature_importance(esg_model, 'esg_score')
            component_scores = self._calculate_component_scores(processed_input, feature_importance)

            return {
                'total_esg_score': esg_score,
                'environmental_score': component_scores['environmental'],
                'social_score': component_scores['social'],
                'governance_score': component_scores['governance'],
                'feature_importance': feature_importance
            }
        except Exception as e:
            self.logger.error(f"Error in ESG score calculation: {str(e)}")
            return {'error': str(e)}

    def _get_feature_importance(self, model, dataset_key):
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.enhanced_model.get_feature_names(dataset_key), model.feature_importances_))
        return {}

    def _calculate_component_scores(self, processed_input, feature_importance):
        env_features = [f for f in feature_importance if 'environment' in f.lower()]
        soc_features = [f for f in feature_importance if 'social' in f.lower()]
        gov_features = [f for f in feature_importance if 'governance' in f.lower()]

        return {
            'environmental': self._weighted_score(processed_input, feature_importance, env_features),
            'social': self._weighted_score(processed_input, feature_importance, soc_features),
            'governance': self._weighted_score(processed_input, feature_importance, gov_features)
        }

    def _weighted_score(self, processed_input, feature_importance, features):
        relevant_importances = [feature_importance[f] for f in features]
        relevant_values = processed_input[0, [self.enhanced_model.get_feature_names('esg_score').index(f) for f in features]]
        return np.dot(relevant_values, relevant_importances) / np.sum(relevant_importances)
    
class InnovativeBusinessModelGenerator:
    def __init__(self, enhanced_model: EnhancedSustainabilityModel):
        self.enhanced_model = enhanced_model
        self.logger = enhanced_model.logger
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
            self.logger.error(f"Error in business model generation: {str(e)}")
            return self._generate_fallback_insights(str(e))

    def _validate_input(self, data: Dict[str, Any]):
        required_fields = ['industry', 'target_market', 'key_resources']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    def _integrate_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        integrated_data = pd.DataFrame()
        
        for dataset_name in ['innovative_startups', 'sdg_indicator', 'world_bank']:
            if dataset_name in self.enhanced_model.data:
                dataset = self.enhanced_model.data[dataset_name]
                if dataset is not None and not dataset.empty:
                    if dataset_name == 'innovative_startups':
                        dataset = dataset[dataset['Industry'] == input_data['industry']]
                    elif dataset_name == 'sdg_indicator':
                        dataset = dataset[dataset['Indicator Code'] == '17.16.1']
                    elif dataset_name == 'world_bank':
                        dataset = dataset[dataset['Country Name'] == input_data.get('country', 'World')]
                    
                    integrated_data = pd.concat([integrated_data, dataset], axis=1)
        
        integrated_data['input_description'] = input_data.get('description', '')
        return integrated_data

    def _analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        market_size = data['Last Valuation (Billion $)'].sum() if 'Last Valuation (Billion $)' in data.columns else 0
        top_companies = data.nlargest(5, 'Last Valuation (Billion $)')['Company'].tolist() if 'Company' in data.columns else []
        avg_valuation = data['Last Valuation (Billion $)'].mean() if 'Last Valuation (Billion $)' in data.columns else 0
        
        return {
            'market_size': market_size,
            'top_companies': top_companies,
            'average_valuation': avg_valuation
        }

    def _calculate_sustainability_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        company_data = {}
        for column in ['Full Time Employees', 'Controversy Level', 'aqi_value', 'energy_per_capita', 'renewable_electricity', 'gdp']:
            if column in data.columns:
                company_data[column] = data[column].mean() if pd.api.types.is_numeric_dtype(data[column]) else data[column].mode()[0]
            else:
                company_data[column] = 0
        
        esg_calculator = ESGScoreCalculator(self.enhanced_model)
        return esg_calculator.calculate_esg_score(company_data)

    def _calculate_innovation_score(self, data: pd.DataFrame) -> float:
        if 'Year Joined' in data.columns and 'Last Valuation (Billion $)' in data.columns:
            age = datetime.now().year - data['Year Joined'].mean()
            valuation = data['Last Valuation (Billion $)'].mean()
            return (valuation / age) * 100 if age > 0 else 0
        return 0

    def _forecast_future_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        if self.enhanced_model.prophet_model is not None:
            future_dates = self.enhanced_model.prophet_model.make_future_dataframe(periods=5, freq='Y')
            forecast = self.enhanced_model.prophet_model.predict(future_dates)
            return {'urban_population_forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_dict('records')}
        return {}

    def _generate_business_model_content(self, input_data: Dict[str, Any], market_insights: Dict[str, Any], 
                                         sustainability_metrics: Dict[str, float], innovation_score: float, 
                                         future_trends: Dict[str, Any]) -> Dict[str, Any]:
        # Use the trained model from EnhancedSustainabilityModel for text analysis
        sentiment_score = self.analyze_text(input_data.get('description', ''))['sentiment_score']
        
        value_proposition = f"Innovative {input_data['industry']} solution addressing {input_data['target_market']} needs with a focus on sustainability (ESG Score: {sustainability_metrics.get('total_esg_score', 0):.2f})"
        
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
        
        sustainability_impact = f"Potential to reduce industry environmental impact by {sustainability_metrics.get('environmental_score', 0)*10:.1f}%"
        
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
            "scalability_plan": scalability_plan,
            "sentiment_score": sentiment_score
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

    def analyze_text(self, text: str) -> Dict[str, float]:
        inputs = self.nlp_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.nlp_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        
        text_analysis_model = self.enhanced_model.get_model('text_analysis')
        if text_analysis_model is not None:
            analysis_result = text_analysis_model.predict(embeddings)
            return {'sentiment_score': float(analysis_result[0])}
        else:
            return {'sentiment_score': 0.0}  # Fallback if no model is available
        


# Main execution
if __name__ == "__main__":
    enhanced_model = EnhancedSustainabilityModel()
    enhanced_model.load_data()
    enhanced_model.preprocess_data()
    enhanced_model.train_models()
    enhanced_model.train_time_series_model()

    # Initialize other classes with the EnhancedSustainabilityModel
    text_generator = TextGenerator(enhanced_model)
    image_generator = ImageGenerator(enhanced_model)
    predictive_analytics = PredictiveAnalytics(enhanced_model)
    environmental_impact_analyzer = EnvironmentalImpactAnalyzer(enhanced_model)
    esg_score_calculator = ESGScoreCalculator(enhanced_model)
    business_model_generator = InnovativeBusinessModelGenerator(enhanced_model)

    # Example usage
    company_name = "Example Corp"
    sustainability_report = enhanced_model.generate_sustainability_report(company_name)
    print(f"Sustainability Report for {company_name}:")
    print(sustainability_report)