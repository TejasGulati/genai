import os
import logging
from datetime import datetime
import random
import time
import concurrent.futures
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Counter, Dict, List, Any
import numpy as np
from diffusers import StableDiffusionPipeline
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
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

from xgboost import XGBClassifier, XGBRegressor
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
        self.cv_scores = {}
        self.time_series_forecasts = None
        
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
                    
                        # Define relevant indicators
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
                        
                        # Filter for relevant indicators
                        df = df[df['Indicator Name'].isin(relevant_indicators)]
                        
                        # Identify year columns
                        year_columns = [col for col in df.columns if col.isdigit()]
                        
                        # Filter for entries with at least 63 data points
                        df['data_points'] = df[year_columns].notna().sum(axis=1)
                        df = df[df['data_points'] >= 64]
                        df = df.drop('data_points', axis=1)
                        
                        # Melt the DataFrame to convert years to a single column
                        id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
                        df = pd.melt(df, id_vars=id_vars, value_vars=year_columns, var_name='year', value_name='value')
                        
                        # Convert year and value to numeric and drop NaNs
                        df['year'] = pd.to_numeric(df['year'], errors='coerce')
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        df = df.dropna(subset=['year', 'value'])
                        
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
                self.logger.warning(f"Skipping {key} dataset due to missing data or target variable")
                continue

            model_filename = f"{key}_model.joblib"
            if os.path.exists(model_filename):
                self.logger.info(f"Loading existing model for {key} dataset")
                self.models[key] = self.load_model(key)
                continue

            self.logger.info(f"Starting model training for {key} dataset")
            self.logger.info(f"Shape of {key} dataset: {self.data[key].shape}")

            y = self.data[key][self.target_variables[key]]
            mask = ~y.isna()
            X = self.data[key][mask]
            y = y[mask]

            if len(y) == 0:
                self.logger.warning(f"No valid target values for {key} dataset. Skipping.")
                continue

            # Use a smaller subset of data for faster training
            if len(X) > 10000:
                X, _, y, _ = train_test_split(X, y, train_size=10000, random_state=42)

            X = self.preprocessors[key].fit_transform(X)
            is_classification = y.dtype == 'object' or len(np.unique(y)) < 10

            if is_classification:
                le = LabelEncoder()
                y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if is_classification:
                models = {
                    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
                    'LightGBM': lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1),
                    'XGBoost': XGBClassifier(random_state=42, scale_pos_weight=1),
                    'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced')
                }
                param_grids = {
                    'RandomForest': {
                        'n_estimators': [100, 200],
                        'max_depth': [None, 20],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'LightGBM': {
                        'n_estimators': [100, 200],
                        'max_depth': [-1, 20],
                        'num_leaves': [31, 50],
                        'reg_alpha': [0, 0.1, 0.5],
                        'reg_lambda': [0, 0.1, 0.5]
                    },
                    'XGBoost': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6],
                        'learning_rate': [0.01, 0.1],
                        'reg_alpha': [0, 0.1, 0.5],
                        'reg_lambda': [0, 0.1, 0.5]
                    },
                    'LogisticRegression': {
                        'C': [0.1, 1, 10],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga']
                    }
                }
                scoring = 'balanced_accuracy'
            else:
                models = {
                    'RandomForest': RandomForestRegressor(random_state=42),
                    'LightGBM': lgb.LGBMRegressor(random_state=42, verbosity=-1),
                    'XGBoost': XGBRegressor(random_state=42),
                    'Lasso': LassoCV(random_state=42),
                    'Ridge': RidgeCV()
                }
                param_grids = {
                    'RandomForest': {
                        'n_estimators': [100, 200],
                        'max_depth': [None, 20],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'LightGBM': {
                        'n_estimators': [100, 200],
                        'max_depth': [-1, 20],
                        'num_leaves': [31, 50],
                        'reg_alpha': [0, 0.1, 0.5],
                        'reg_lambda': [0, 0.1, 0.5]
                    },
                    'XGBoost': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6],
                        'learning_rate': [0.01, 0.1],
                        'reg_alpha': [0, 0.1, 0.5],
                        'reg_lambda': [0, 0.1, 0.5]
                    },
                    'Lasso': {'eps': [0.1, 0.01, 0.001]},
                    'Ridge': {'alphas': [0.1, 1, 10]}
                }
                scoring = 'neg_mean_squared_error'

            best_models = {}
            for model_name, model in models.items():
                self.logger.info(f"Training and tuning {model_name} for {key} dataset")
                grid_search = RandomizedSearchCV(
                    model, 
                    param_grids[model_name], 
                    cv=5,  # 5-fold cross-validation
                    scoring=scoring, 
                    n_iter=20,
                    n_jobs=-1, 
                    random_state=42
                )
                grid_search.fit(X_train, y_train)
                best_models[model_name] = grid_search.best_estimator_
                self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                self.logger.info(f"Best cross-validation score for {model_name}: {grid_search.best_score_}")
                
                # Perform additional cross-validation
                cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring=scoring)
                self.cv_scores[f"{key}_{model_name}"] = cv_scores
                self.logger.info(f"Cross-validation scores for {model_name}: {cv_scores}")
                self.logger.info(f"Mean CV score: {np.mean(cv_scores)}, Std CV score: {np.std(cv_scores)}")

            if is_classification:
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in best_models.items()],
                    voting='soft'
                )
            else:
                ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in best_models.items()]
                )

            ensemble.fit(X_train, y_train)
        
            y_pred = ensemble.predict(X_test)
            if is_classification:
                accuracy = balanced_accuracy_score(y_test, y_pred)
                self.logger.info(f"{key} - Test set balanced accuracy: {accuracy:.4f}")
            else:
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                self.logger.info(f"{key} - Test set R2: {r2:.4f}, MSE: {mse:.4f}")

            self.models[key] = ensemble
            self.logger.info(f"Best ensemble model for {key} dataset")

            self.explain_model(key, ensemble, X_test)
            self.save_model(key, ensemble)

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

    def load_time_series_forecasts(self):
        forecast_file = 'time_series_forecasts.csv'
        if os.path.exists(forecast_file):
            self.time_series_forecasts = pd.read_csv(forecast_file)
            self.logger.info(f"Loaded time series forecasts from {forecast_file}")
        else:
            self.logger.warning(f"Time series forecast file {forecast_file} not found")

    def train_time_series_model(self):
        forecast_file = 'time_series_forecasts.csv'
        if os.path.exists(forecast_file):
            self.load_time_series_forecasts()
            self.logger.info("Using existing time series forecasts")
            return

        try:
            df = self.data['world_bank']
            df_prophet_all = self._prepare_prophet_data(df)

            min_data_points = 64
            groups = [(country, indicator, group) for (country, indicator), group in df_prophet_all.groupby(['country', 'indicator']) if len(group) >= min_data_points]
            total_combinations = len(groups)
            processed = 0

            self.prophet_models = {}
            self.model_confidence = {}
            all_forecasts = []

            start_time = time.time()

            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._train_and_forecast, country, indicator, group) for country, indicator, group in groups]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            country, indicator, model, confidence, forecast_data = result
                            if country not in self.prophet_models:
                                self.prophet_models[country] = {}
                            self.prophet_models[country][indicator] = model
                            self.model_confidence[(country, indicator)] = confidence
                            all_forecasts.append(forecast_data)
                        
                        processed += 1
                        self._print_progress(processed, total_combinations, start_time)
                    except Exception as e:
                        self.logger.error(f"Error processing future: {str(e)}")

            print("\nTime series modeling completed.")
            self._save_forecasts(all_forecasts)
            self.load_time_series_forecasts()

        except Exception as e:
            self.logger.error(f"Error in training time series model: {str(e)}")
            self.prophet_models = None
            self.model_confidence = {}

    def make_time_series_forecast(self, country, indicator):
        if self.time_series_forecasts is None:
            self.load_time_series_forecasts()

        if self.time_series_forecasts is not None:
            forecast_data = self.time_series_forecasts[
                (self.time_series_forecasts['country'] == country) &
                (self.time_series_forecasts['indicator'] == indicator)
            ]

            if not forecast_data.empty:
                result = {
                    'forecast': [
                        {
                            'ds': f"{year}-01-01",
                            'yhat': forecast_data[f'{year}_yhat'].values[0],
                            'actual': forecast_data[f'{year}_actual'].values[0] if f'{year}_actual' in forecast_data.columns else None
                        }
                        for year in range(2020, 2026)
                    ],
                    'confidence': self.model_confidence.get((country, indicator), 0.8)
                }
                return result

        # If forecast not found in CSV, fall back to training a new model
        try:
            if self.prophet_models is None or country not in self.prophet_models or indicator not in self.prophet_models[country]:
                raise ValueError(f"No trained model found for {country}, {indicator}")

            model = self.prophet_models[country][indicator]
            future_dates = model.make_future_dataframe(periods=2, freq='Y')
            forecast = model.predict(future_dates)

            confidence = self.model_confidence.get((country, indicator), 0)
            
            result = {
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(2).to_dict('records'),
                'confidence': confidence
            }

            return result
        except Exception as e:
            self.logger.error(f"Error in making time series forecast: {str(e)}")
            return {'error': str(e)}

    def _prepare_prophet_data(self, df):
        prophet_data = []
        for _, group in df.groupby(['Country Name', 'Indicator Name']):
            df_prophet = pd.DataFrame({
                'ds': pd.to_datetime(group['year'], format='%Y'),
                'y': pd.to_numeric(group['value'], errors='coerce'),
                'country': group['Country Name'],
                'indicator': group['Indicator Name']
            })
            prophet_data.append(df_prophet)
        return pd.concat(prophet_data, ignore_index=True).dropna().sort_values(['country', 'indicator', 'ds'])


    def _train_and_forecast(self, country, indicator, group):
        try:
            model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            model.fit(group[['ds', 'y']])
            
            df_cv = cross_validation(model, initial='3650 days', period='365 days', horizon='730 days')
            df_p = performance_metrics(df_cv)
            rmse = df_p['rmse'].mean()
            
            confidence = min(1.0, (len(group) / 40) * (1 / (1 + rmse)))
            
            future = model.make_future_dataframe(periods=6, freq='Y')
            forecast = model.predict(future)
            
            forecast_data = forecast[['ds', 'yhat']]
            forecast_data['country'] = country
            forecast_data['indicator'] = indicator
            forecast_data['actual'] = group.set_index('ds')['y']
            forecast_data = forecast_data[(forecast_data['ds'].dt.year >= 2020) & (forecast_data['ds'].dt.year <= 2025)]
            
            return country, indicator, model, confidence, forecast_data
        except Exception:
            return None

    def _print_progress(self, processed, total, start_time):
        progress = (processed / total) * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (processed / total)
        remaining_time = estimated_total_time - elapsed_time
        
        print(f"\rProgress: {progress:.2f}% | Elapsed: {elapsed_time:.2f}s | Remaining: {remaining_time:.2f}s", end="", flush=True)

    def _save_forecasts(self, all_forecasts):
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        combined_forecasts['year'] = combined_forecasts['ds'].dt.year
        pivot_forecasts = combined_forecasts.pivot_table(
            values=['actual', 'yhat'],
            index=['country', 'indicator'],
            columns='year',
            aggfunc='first'
        )
        pivot_forecasts.columns = [f'{col[1]}_{col[0]}' for col in pivot_forecasts.columns]
        pivot_forecasts.reset_index(inplace=True)
        column_order = ['country', 'indicator'] + [f'{year}_{col}' for year in range(2020, 2026) for col in ['actual', 'yhat']]
        pivot_forecasts = pivot_forecasts.reindex(columns=column_order)
        pivot_forecasts.to_csv('time_series_forecasts.csv', index=False)
        print(f"\nForecasts saved to time_series_forecasts.csv")

    

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
            
            # Calculate ESG score
            esg_score = self._calculate_esg_score(esg_data)
            
            report = {
                'company_name': company_name,
                'industry': company_data['industry'],
                'country': company_data['country'],
                'year_founded': company_data['year founded'],
                'size_range': company_data['size range'],
                'esg_scores': {
                    'total': esg_score,
                    'environment': esg_data['Environment Risk Score'],
                    'social': esg_data['Social Risk Score'],
                    'governance': esg_data['Governance Risk Score']
                },
                'sustainability_prediction': predictions['model_predictions']['rf'],
                'key_factors': dict(sorted(predictions['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]),
                'recommendations': self._generate_recommendations(predictions, country_air_quality, country_energy_data, esg_score),
                'country_metrics': {
                    'air_quality': country_air_quality,
                    'energy': {
                        'renewable_electricity': country_energy_data.get('renewable_electricity', 'N/A'),
                        'energy_per_capita': country_energy_data.get('energy_per_capita', 'N/A'),
                        'co2_emissions': country_energy_data.get('co2_emissions', 'N/A')
                    }
                },
                'innovative_potential': self._assess_innovative_potential(company_data, esg_score)
            }
            
            self.logger.info(f"Sustainability report generated for {company_name}")
            return report
        except Exception as e:
            self.logger.error(f"Error in generating sustainability report: {str(e)}")
            return {'error': str(e)}

    def _calculate_esg_score(self, esg_data: pd.Series) -> float:
        weights = {'Environment Risk Score': 0.4, 'Social Risk Score': 0.3, 'Governance Risk Score': 0.3}
        return sum(esg_data[key] * weight for key, weight in weights.items())

    def _assess_innovative_potential(self, company_data: pd.Series, esg_score: float) -> Dict[str, Any]:
        industry_innovativeness = self.data['innovative_startups'][self.data['innovative_startups']['Industry'] == company_data['industry']]['Last Valuation (Billion $)'].mean()
        company_age = datetime.now().year - company_data['year founded']
        
        innovation_score = (industry_innovativeness / company_age) * (esg_score / 100)
        
        return {
            'score': innovation_score,
            'industry_comparison': 'Above average' if innovation_score > industry_innovativeness else 'Below average',
            'potential_areas': self._generate_innovation_areas(company_data['industry'], esg_score)
        }

    def _generate_innovation_areas(self, industry: str, esg_score: float) -> List[str]:
        areas = [
            f"AI-driven {industry} solutions",
            "Sustainable supply chain optimization",
            "Circular economy business models"
        ]
        if esg_score > 80:
            areas.append("ESG-focused product development")
        return areas

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
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.logger = enhanced_model.logger

    def generate(self, prompt: str, max_length: int = 100) -> str:
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
            
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Error in text generation: {str(e)}")
            return f"Error in text generation: {str(e)}"

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
        air_quality_data = self.enhanced_model.data.get('air_quality')
        if air_quality_data is None:
            return {'aqi_value': None, 'co_aqi_value': None, 'no2_aqi_value': None, 'pm2.5_aqi_value': None}
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
        energy_data = self.enhanced_model.data.get('energy')
        if energy_data is None:
            return {'energy_per_capita': None, 'renewable_electricity': None}
        country_data = energy_data[(energy_data['country'] == country) & (energy_data['year'] == year)]
        if country_data.empty:
            return {'energy_per_capita': None, 'renewable_electricity': None}
        return {
            'energy_per_capita': country_data['energy_per_capita'].values[0],
            'renewable_electricity': country_data['renewable_electricity'].values[0]
        }

    def _get_climate_indicators(self, country: str, year: int) -> Dict[str, float]:
        world_bank_data = self.enhanced_model.data.get('world_bank')
        if world_bank_data is None:
            return {'urban_population_pct': None}
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
    predictive_analytics = PredictiveAnalytics(enhanced_model)
    environmental_impact_analyzer = EnvironmentalImpactAnalyzer(enhanced_model)
    business_model_generator = InnovativeBusinessModelGenerator(enhanced_model)

    # Example usage
    company_name = "Example Corp"
    sustainability_report = enhanced_model.generate_sustainability_report(company_name)
    print(f"Sustainability Report for {company_name}:")
    print(sustainability_report)