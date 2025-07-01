import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Time series and ML libraries
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

from config import settings
from feature_engineering import feature_pipeline

logger = logging.getLogger(__name__)

def clean_for_json(data: Any) -> Any:
    """Clean data for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(item) for item in data]
    elif isinstance(data, (int, float)):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return float(data)
    elif isinstance(data, np.ndarray):
        return clean_for_json(data.tolist())
    elif pd.isna(data):
        return None
    return data

@dataclass
class PredictionMetadata:
    """Metadata for predictive models"""
    model_name: str
    model_type: str
    version: str
    training_date: datetime
    training_samples: int
    features_count: int
    prediction_horizon: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    model_path: str

class TimeSeriesForecaster:
    """Time series forecasting using Prophet and ARIMA"""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.prophet_model = None
        self.arima_model = None
        self.is_fitted = False
        self.seasonal_periods = None
        
    def fit(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Fit both Prophet and ARIMA models"""
        if df.empty:
            raise ValueError("Empty DataFrame provided for training")
        
        training_results = {}
        
        # Prepare data for Prophet
        prophet_data = df.reset_index()
        prophet_data = prophet_data.rename(columns={
            prophet_data.columns[0]: 'ds',  # timestamp column
            target_column: 'y'
        })
        
        # Remove any missing values
        prophet_data = prophet_data.dropna(subset=['y'])
        
        if len(prophet_data) < 10:
            raise ValueError(f"Insufficient data for training: {len(prophet_data)} samples")
        
        # Train Prophet model
        try:
            self.prophet_model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.8
            )
            
            # Add custom seasonalities for monitoring data
            self.prophet_model.add_seasonality(
                name='business_hours',
                period=1,
                fourier_order=3,
                condition_name='is_business_hour'
            )
            
            # Add business hour indicator
            prophet_data['is_business_hour'] = (
                (prophet_data['ds'].dt.hour >= 9) & 
                (prophet_data['ds'].dt.hour <= 17) & 
                (prophet_data['ds'].dt.dayofweek < 5)
            )
            
            self.prophet_model.fit(prophet_data)
            training_results['prophet'] = "success"
            
        except Exception as e:
            logger.warning(f"Prophet training failed for {self.metric_name}: {e}")
            training_results['prophet'] = f"failed: {e}"
        
        # Train ARIMA model
        try:
            # Prepare data for ARIMA (use original time series)
            ts_data = df[target_column].dropna()
            
            if len(ts_data) >= 50:  # Need sufficient data for ARIMA
                # Find optimal ARIMA parameters
                best_aic = np.inf
                best_order = (1, 1, 1)
                
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model = ARIMA(ts_data, order=(p, d, q))
                                fitted = model.fit()
                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                # Train final ARIMA model
                self.arima_model = ARIMA(ts_data, order=best_order)
                self.arima_fitted = self.arima_model.fit()
                training_results['arima'] = f"success (order: {best_order})"
                
            else:
                training_results['arima'] = "insufficient_data"
                
        except Exception as e:
            logger.warning(f"ARIMA training failed for {self.metric_name}: {e}")
            training_results['arima'] = f"failed: {e}"
        
        self.is_fitted = True
        return training_results
    
    def predict(self, periods: int = 24, frequency: str = 'H') -> Dict[str, pd.DataFrame]:
        """Generate predictions from both models"""
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        predictions = {}
        
        # Prophet predictions
        if self.prophet_model:
            try:
                future = self.prophet_model.make_future_dataframe(
                    periods=periods, 
                    freq=frequency
                )
                
                # Add business hour indicator for future dates
                future['is_business_hour'] = (
                    (future['ds'].dt.hour >= 9) & 
                    (future['ds'].dt.hour <= 17) & 
                    (future['ds'].dt.dayofweek < 5)
                )
                
                forecast = self.prophet_model.predict(future)
                predictions['prophet'] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                
            except Exception as e:
                logger.error(f"Prophet prediction failed: {e}")
        
        # ARIMA predictions
        if hasattr(self, 'arima_fitted'):
            try:
                arima_forecast = self.arima_fitted.forecast(steps=periods)
                arima_conf_int = self.arima_fitted.get_forecast(steps=periods).conf_int()
                
                # Create timestamp index for ARIMA predictions
                last_timestamp = self.arima_fitted.data.dates[-1]
                future_dates = pd.date_range(
                    start=last_timestamp + pd.Timedelta(hours=1),
                    periods=periods,
                    freq=frequency
                )
                
                predictions['arima'] = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': arima_forecast.values,
                    'yhat_lower': arima_conf_int.iloc[:, 0].values,
                    'yhat_upper': arima_conf_int.iloc[:, 1].values
                })
                
            except Exception as e:
                logger.error(f"ARIMA prediction failed: {e}")
        
        return predictions
    
    def get_forecast_accuracy(self, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        if test_data.empty:
            return {}
        
        accuracy_metrics = {}
        
        # Get predictions for test period
        test_periods = len(test_data)
        predictions = self.predict(periods=test_periods)
        
        actual_values = test_data[target_column].values
        
        # Prophet accuracy
        if 'prophet' in predictions:
            prophet_pred = predictions['prophet']['yhat'].values[:len(actual_values)]
            if len(prophet_pred) == len(actual_values):
                accuracy_metrics['prophet_mae'] = mean_absolute_error(actual_values, prophet_pred)
                accuracy_metrics['prophet_rmse'] = np.sqrt(mean_squared_error(actual_values, prophet_pred))
                accuracy_metrics['prophet_mape'] = np.mean(np.abs((actual_values - prophet_pred) / (actual_values + 1e-8))) * 100
        
        # ARIMA accuracy
        if 'arima' in predictions:
            arima_pred = predictions['arima']['yhat'].values[:len(actual_values)]
            if len(arima_pred) == len(actual_values):
                accuracy_metrics['arima_mae'] = mean_absolute_error(actual_values, arima_pred)
                accuracy_metrics['arima_rmse'] = np.sqrt(mean_squared_error(actual_values, arima_pred))
                accuracy_metrics['arima_mape'] = np.mean(np.abs((actual_values - arima_pred) / (actual_values + 1e-8))) * 100
        
        return clean_for_json(accuracy_metrics)

class FailurePredictionModel:
    """Binary classification model for failure prediction"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train failure prediction model"""
        if X.empty or len(y) == 0:
            raise ValueError("Empty training data provided")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle class imbalance
        from imblearn.over_sampling import SMOTE
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"Applied SMOTE: {len(X)} -> {len(X_resampled)} samples")
        except:
            X_resampled, y_resampled = X, y
            logger.warning("SMOTE failed, using original data")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        self.is_fitted = True
        
        return {
            'performance_metrics': clean_for_json(metrics),
            'feature_importance': clean_for_json(feature_importance),
            'training_samples': len(X_resampled),
            'test_samples': len(X_test)
        }
    
    def predict_failure_probability(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predict failure probability for new data"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Align features
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0.0)
        
        # Get predictions
        failure_proba = self.model.predict_proba(X_aligned)[:, 1]
        failure_pred = self.model.predict(X_aligned)
        
        # Calculate time to failure estimate
        time_to_failure = self._estimate_time_to_failure(failure_proba)
        
        # Risk assessment
        risk_levels = np.where(
            failure_proba >= 0.8, 'CRITICAL',
            np.where(failure_proba >= 0.6, 'HIGH',
            np.where(failure_proba >= 0.4, 'MEDIUM', 'LOW'))
        )
        
        results = {
            'failure_probability': failure_proba.tolist(),
            'failure_prediction': failure_pred.tolist(),
            'risk_level': risk_levels.tolist(),
            'time_to_failure_hours': time_to_failure.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return clean_for_json(results)
    
    def _estimate_time_to_failure(self, failure_proba: np.ndarray) -> np.ndarray:
        """Estimate time to failure based on probability"""
        # Simple heuristic: higher probability = shorter time to failure
        # This could be enhanced with survival analysis
        max_time_hours = 72  # Maximum prediction horizon
        time_to_failure = max_time_hours * (1 - failure_proba)
        
        # Ensure minimum time of 1 hour for high probabilities
        time_to_failure = np.maximum(time_to_failure, 1.0)
        
        return time_to_failure

class PredictiveModelManager:
    """Main manager for all predictive models"""
    
    def __init__(self):
        self.forecasters = {}
        self.failure_model = FailurePredictionModel()
        self.models_dir = settings.ai.ml_storage_path / "predictive_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    async def train_forecasting_models(
        self, 
        metrics_df: pd.DataFrame,
        target_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Train time series forecasting models for key metrics"""
        if metrics_df.empty:
            raise ValueError("No data provided for training")
        
        if target_metrics is None:
            target_metrics = settings.monitoring.key_metrics
        
        results = {}
        
        for metric in target_metrics:
            if metric in metrics_df.columns:
                logger.info(f"Training forecasting models for {metric}")
                
                # Create forecaster
                forecaster = TimeSeriesForecaster(metric)
                
                # Split data for validation
                split_point = int(len(metrics_df) * 0.8)
                train_data = metrics_df.iloc[:split_point]
                test_data = metrics_df.iloc[split_point:]
                
                try:
                    # Train models
                    training_results = forecaster.fit(train_data, metric)
                    
                    # Evaluate accuracy
                    accuracy_metrics = forecaster.get_forecast_accuracy(test_data, metric)
                    
                    # Store forecaster
                    self.forecasters[metric] = forecaster
                    
                    results[metric] = {
                        'training_status': training_results,
                        'accuracy_metrics': accuracy_metrics,
                        'training_samples': len(train_data),
                        'test_samples': len(test_data)
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to train forecaster for {metric}: {e}")
                    results[metric] = {'error': str(e)}
            else:
                logger.warning(f"Metric {metric} not found in data")
                results[metric] = {'error': 'metric_not_found'}
        
        # Save models
        self._save_forecasting_models()
        
        return clean_for_json(results)
    
    def train_failure_prediction_model(
        self, 
        features_df: pd.DataFrame, 
        alerts_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train failure prediction model"""
        if features_df.empty:
            raise ValueError("No features provided for training")
        
        # Create failure labels from alerts
        failure_labels = self._create_failure_labels(features_df, alerts_df)
        
        # Use feature engineering pipeline
        if not feature_pipeline.is_fitted:
            processed_features = feature_pipeline.fit_transform(features_df)
        else:
            processed_features = feature_pipeline.transform(features_df)
        
        # Align labels with processed features
        common_index = processed_features.index.intersection(failure_labels.index)
        processed_features = processed_features.loc[common_index]
        failure_labels = failure_labels.loc[common_index]
        
        if len(processed_features) == 0:
            raise ValueError("No aligned data for training")
        
        # Train model
        training_results = self.failure_model.fit(processed_features, failure_labels)
        
        # Save model
        self._save_failure_model()
        
        return training_results
    
    def _create_failure_labels(
        self, 
        features_df: pd.DataFrame, 
        alerts_df: pd.DataFrame
    ) -> pd.Series:
        """Create failure labels from alert data"""
        # Initialize all as non-failure
        failure_labels = pd.Series(0, index=features_df.index, name='failure')
        
        if not alerts_df.empty:
            # Mark periods before alerts as failure precursors
            for _, alert in alerts_df.iterrows():
                alert_time = pd.to_datetime(alert['alert_time'])
                
                # Mark 2 hours before alert as failure precursor
                start_time = alert_time - timedelta(hours=2)
                
                mask = (
                    (features_df.index >= start_time) & 
                    (features_df.index <= alert_time)
                )
                failure_labels.loc[mask] = 1
        
        logger.info(f"Created failure labels: {failure_labels.sum()} failures out of {len(failure_labels)} samples")
        return failure_labels
    
    def generate_forecasts(
        self, 
        metrics: List[str] = None,
        horizon_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate forecasts for specified metrics"""
        if metrics is None:
            metrics = list(self.forecasters.keys())
        
        forecasts = {}
        
        for metric in metrics:
            if metric in self.forecasters:
                try:
                    predictions = self.forecasters[metric].predict(
                        periods=horizon_hours, 
                        frequency='H'
                    )
                    
                    # Process predictions for JSON serialization
                    processed_predictions = {}
                    for model_name, pred_df in predictions.items():
                        processed_predictions[model_name] = {
                            'timestamps': pred_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                            'values': pred_df['yhat'].tolist(),
                            'lower_bound': pred_df['yhat_lower'].tolist(),
                            'upper_bound': pred_df['yhat_upper'].tolist()
                        }
                    
                    forecasts[metric] = processed_predictions
                    
                except Exception as e:
                    logger.error(f"Failed to generate forecast for {metric}: {e}")
                    forecasts[metric] = {'error': str(e)}
        
        return {
            'forecasts': forecasts,
            'horizon_hours': horizon_hours,
            'generated_at': datetime.now().isoformat()
        }
    
    def predict_failures(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict failures using recent monitoring data"""
        if not self.failure_model.is_fitted:
            raise ValueError("Failure prediction model not trained")
        
        # Process features using same pipeline
        processed_features = feature_pipeline.transform(recent_data)
        
        # Get failure predictions
        failure_results = self.failure_model.predict_failure_probability(processed_features)
        
        # Add recommendations based on predictions
        recommendations = self._generate_failure_recommendations(failure_results)
        
        failure_results['recommendations'] = recommendations
        
        return failure_results
    
    def _generate_failure_recommendations(self, failure_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on failure predictions"""
        recommendations = []
        
        failure_probs = failure_results['failure_probability']
        risk_levels = failure_results['risk_level']
        time_to_failure = failure_results['time_to_failure_hours']
        
        for i, (prob, risk, ttf) in enumerate(zip(failure_probs, risk_levels, time_to_failure)):
            if risk == 'CRITICAL':
                recommendations.append({
                    'priority': 'CRITICAL',
                    'action': 'Immediate intervention required',
                    'description': f'System failure predicted within {ttf:.1f} hours with {prob:.1%} confidence',
                    'suggested_actions': [
                        'Check system logs immediately',
                        'Verify hardware status',
                        'Consider immediate maintenance',
                        'Alert on-call team'
                    ]
                })
            elif risk == 'HIGH':
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'Schedule maintenance within 24 hours',
                    'description': f'Potential failure in {ttf:.1f} hours with {prob:.1%} confidence',
                    'suggested_actions': [
                        'Schedule proactive maintenance',
                        'Monitor closely',
                        'Prepare backup systems',
                        'Review recent changes'
                    ]
                })
            elif risk == 'MEDIUM':
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'Monitor and plan maintenance',
                    'description': f'Moderate risk detected with {prob:.1%} failure probability',
                    'suggested_actions': [
                        'Increase monitoring frequency',
                        'Plan maintenance window',
                        'Review performance trends'
                    ]
                })
        
        return recommendations
    
    def _save_forecasting_models(self):
        """Save trained forecasting models"""
        models_path = self.models_dir / "forecasters.joblib"
        joblib.dump(self.forecasters, models_path)
        logger.info(f"Forecasting models saved: {models_path}")
    
    def _save_failure_model(self):
        """Save trained failure prediction model"""
        model_path = self.models_dir / "failure_predictor.joblib"
        joblib.dump(self.failure_model, model_path)
        logger.info(f"Failure prediction model saved: {model_path}")
    
    def load_models(self) -> bool:
        """Load saved models"""
        try:
            # Load forecasting models
            forecasters_path = self.models_dir / "forecasters.joblib"
            if forecasters_path.exists():
                self.forecasters = joblib.load(forecasters_path)
                logger.info("Forecasting models loaded")
            
            # Load failure prediction model
            failure_path = self.models_dir / "failure_predictor.joblib"
            if failure_path.exists():
                self.failure_model = joblib.load(failure_path)
                logger.info("Failure prediction model loaded")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all predictive models"""
        status = {
            'forecasting_models': {
                'count': len(self.forecasters),
                'metrics': list(self.forecasters.keys())
            },
            'failure_prediction': {
                'trained': self.failure_model.is_fitted,
                'features_count': len(self.failure_model.feature_names) if self.failure_model.is_fitted else 0
            },
            'feature_pipeline': {
                'fitted': feature_pipeline.is_fitted,
                'features_count': len(feature_pipeline.feature_names) if feature_pipeline.is_fitted else 0
            }
        }
        
        return status

# Global predictive model manager
predictive_manager = PredictiveModelManager()



