
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict

# Add Prophet import here
from prophet import Prophet

from predictive_models import (
    predictive_manager, 
    TimeSeriesForecaster, 
    PredictionMetadata,
    clean_for_json
)
from config import settings
from data_collector import DataCollectionManager

logger = logging.getLogger(__name__)

@dataclass
class MetricProfile:
    """Profile for each metric type with specific forecasting parameters"""
    name: str
    data_type: str
    unit: str
    seasonal_patterns: List[str]
    prophet_config: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    accuracy_targets: Dict[str, float]

class MetricAnalyzer:
    """Analyzes metric characteristics for optimal forecasting"""
    
    def __init__(self):
        self.metric_profiles = {}
        
    def analyze_metric_characteristics(self, df: pd.DataFrame, metric_name: str) -> MetricProfile:
        """Analyze metric to determine optimal forecasting approach"""
        
        if metric_name not in df.columns:
            raise ValueError(f"Metric {metric_name} not found in data")
        
        data = df[metric_name].dropna()
        
        if len(data) == 0:
            raise ValueError(f"No valid data for metric {metric_name}")
        
        # Basic statistics
        mean_val = data.mean()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()
        
        # Determine metric characteristics based on name and data patterns
        profile = self._create_metric_profile(metric_name, data, mean_val, std_val, min_val, max_val)
        
        self.metric_profiles[metric_name] = profile
        return profile
    
    def _create_metric_profile(self, metric_name: str, data: pd.Series, 
                             mean_val: float, std_val: float, min_val: float, max_val: float) -> MetricProfile:
        """Create optimized profile for specific metric types"""
        
        # CPU Utilization (percentage-based)
        if 'cpu_util' in metric_name.lower() and not 'load' in metric_name.lower():
            return MetricProfile(
                name=metric_name,
                data_type='percentage',
                unit='%',
                seasonal_patterns=['business_hours', 'daily', 'weekly'],
                prophet_config={
                    'growth': 'logistic',
                    'daily_seasonality': True,
                    'weekly_seasonality': True,
                    'yearly_seasonality': False,
                    'changepoint_prior_scale': 0.001,
                    'seasonality_prior_scale': 15.0,
                    'interval_width': 0.85,
                    'cap': 100.0,
                    'floor': 0.0
                },
                preprocessing_config={
                    'outlier_method': 'iqr',
                    'outlier_factor': 2.0,
                    'smooth_spikes': True,
                    'fill_method': 'interpolate'
                },
                accuracy_targets={'mape': 15.0, 'mae': 5.0}
            )
        
        # CPU Load Average
        elif 'cpu_load' in metric_name.lower():
            is_low_load = mean_val < 2.0 and max_val < 5.0
            
            return MetricProfile(
                name=metric_name,
                data_type='load_average',
                unit='none',
                seasonal_patterns=['business_hours', 'daily', 'weekly'],
                prophet_config={
                    'growth': 'linear' if is_low_load else 'logistic',
                    'daily_seasonality': True,
                    'weekly_seasonality': True,
                    'yearly_seasonality': False,
                    'changepoint_prior_scale': 0.05 if is_low_load else 0.01,
                    'seasonality_prior_scale': 8.0,
                    'interval_width': 0.80,
                    'cap': max(16.0, max_val * 1.5) if not is_low_load else None,
                    'floor': 0.0
                },
                preprocessing_config={
                    'outlier_method': 'percentile',
                    'outlier_percentile': 95,
                    'smooth_spikes': True,
                    'scale_factor': 1000 if max_val > 1000 else (100 if max_val > 100 else 1)
                },
                accuracy_targets={'mape': 25.0, 'mae': mean_val * 0.3}
            )
        
        # Memory Utilization
        elif 'memory' in metric_name.lower() and ('util' in metric_name.lower() or 'pavailable' in metric_name.lower()):
            return MetricProfile(
                name=metric_name,
                data_type='percentage',
                unit='%',
                seasonal_patterns=['business_hours', 'daily'],
                prophet_config={
                    'growth': 'logistic',
                    'daily_seasonality': True,
                    'weekly_seasonality': True,
                    'yearly_seasonality': False,
                    'changepoint_prior_scale': 0.005,
                    'seasonality_prior_scale': 12.0,
                    'interval_width': 0.80,
                    'cap': 100.0,
                    'floor': 0.0
                },
                preprocessing_config={
                    'outlier_method': 'iqr',
                    'outlier_factor': 1.5,
                    'smooth_spikes': False,
                    'fill_method': 'forward_fill'
                },
                accuracy_targets={'mape': 12.0, 'mae': 3.0}
            )
        
        # Network Metrics
        elif 'net_if' in metric_name.lower():
            return MetricProfile(
                name=metric_name,
                data_type='network_traffic',
                unit='bps',
                seasonal_patterns=['business_hours', 'daily'],
                prophet_config={
                    'growth': 'linear',
                    'daily_seasonality': True,
                    'weekly_seasonality': True,
                    'yearly_seasonality': False,
                    'changepoint_prior_scale': 0.08,
                    'seasonality_prior_scale': 5.0,
                    'interval_width': 0.85
                },
                preprocessing_config={
                    'outlier_method': 'percentile',
                    'outlier_percentile': 98,
                    'smooth_spikes': True,
                    'remove_zero_variance': True
                },
                accuracy_targets={'mape': 30.0, 'mae': std_val}
            )
        
        # Default profile for unknown metrics
        else:
            return MetricProfile(
                name=metric_name,
                data_type='generic',
                unit='unknown',
                seasonal_patterns=['daily'],
                prophet_config={
                    'growth': 'linear',
                    'daily_seasonality': True,
                    'weekly_seasonality': True,
                    'yearly_seasonality': False,
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10.0,
                    'interval_width': 0.80
                },
                preprocessing_config={
                    'outlier_method': 'iqr',
                    'outlier_factor': 2.0,
                    'smooth_spikes': True
                },
                accuracy_targets={'mape': 25.0, 'mae': std_val}
            )

class EnhancedTimeSeriesForecaster(TimeSeriesForecaster):
    """Enhanced forecaster with metric-specific optimizations"""
    
    def __init__(self, metric_name: str, profile: MetricProfile):
        super().__init__(metric_name)
        self.profile = profile
        self.data_quality_score = 0.0
        
    def fit(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Enhanced fit with metric-specific preprocessing and configuration"""
        
        if df.empty:
            raise ValueError("Empty DataFrame provided for training")
        
        # Preprocess data according to metric profile
        processed_df = self._preprocess_data(df, target_column)
        
        # Calculate data quality score
        self.data_quality_score = self._calculate_data_quality(processed_df, target_column)
        
        # Prepare data for Prophet with metric-specific configuration
        prophet_data = self._prepare_prophet_data(processed_df, target_column)
        
        if len(prophet_data) < 20:
            raise ValueError(f"Insufficient data after preprocessing: {len(prophet_data)} samples")
        
        training_results = {}
        
        # Configure Prophet model based on metric profile
        try:
            prophet_config = self.profile.prophet_config.copy()
            
            # Create Prophet model with metric-specific settings
            self.prophet_model = Prophet(
                growth=prophet_config.get('growth', 'linear'),
                daily_seasonality=prophet_config.get('daily_seasonality', True),
                weekly_seasonality=prophet_config.get('weekly_seasonality', True),
                yearly_seasonality=prophet_config.get('yearly_seasonality', False),
                changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=prophet_config.get('seasonality_prior_scale', 10.0),
                interval_width=prophet_config.get('interval_width', 0.80)
            )
            
            # Store configuration for later use
            self.prophet_config_used = prophet_config.copy()
            
            # Add capacity constraints if specified
            if 'cap' in prophet_config and prophet_config['cap'] is not None:
                prophet_data['cap'] = prophet_config['cap']
            if 'floor' in prophet_config:
                prophet_data['floor'] = prophet_config['floor']
            
            # Add custom seasonalities based on metric profile
            for pattern in self.profile.seasonal_patterns:
                if pattern == 'business_hours':
                    self.prophet_model.add_seasonality(
                        name='business_hours',
                        period=1,
                        fourier_order=3,
                        condition_name='is_business_hour'
                    )
                    prophet_data['is_business_hour'] = (
                        (prophet_data['ds'].dt.hour >= 9) & 
                        (prophet_data['ds'].dt.hour <= 17) & 
                        (prophet_data['ds'].dt.dayofweek < 5)
                    )
            
            # Store the complete training data BEFORE fitting
            self.prophet_training_data = prophet_data.copy()
            logger.info(f"Stored training data: {len(self.prophet_training_data)} samples with columns: {list(self.prophet_training_data.columns)}")
            
            # Fit the model
            logger.info(f"Fitting Prophet model for {self.metric_name}...")
            self.prophet_model.fit(prophet_data)
            
            # Verify the model is actually fitted
            if not hasattr(self.prophet_model, 'params') or self.prophet_model.params is None:
                raise Exception("Prophet model fitting failed - no parameters generated")
            
            logger.info(f"Prophet model successfully fitted for {self.metric_name}")
            
            training_results['prophet'] = {
                'status': 'success',
                'data_quality_score': self.data_quality_score,
                'samples_used': len(prophet_data),
                'profile_type': self.profile.data_type
            }
            
        except Exception as e:
            logger.error(f"Prophet training failed for {self.metric_name}: {e}")
            training_results['prophet'] = {'status': 'failed', 'error': str(e)}
            self.prophet_model = None
        
        # Enhanced ARIMA training with better error handling
        try:
            ts_data = processed_df[target_column].dropna()
            
            if len(ts_data) >= 30:
                # Use metric-specific ARIMA parameters
                best_order = self._find_optimal_arima_order(ts_data)
                
                from statsmodels.tsa.arima.model import ARIMA
                self.arima_model = ARIMA(ts_data, order=best_order)
                self.arima_fitted = self.arima_model.fit()
                
                training_results['arima'] = {
                    'status': 'success',
                    'order': best_order,
                    'aic': self.arima_fitted.aic
                }
            else:
                training_results['arima'] = {'status': 'insufficient_data'}
                
        except Exception as e:
            logger.error(f"ARIMA training failed for {self.metric_name}: {e}")
            training_results['arima'] = {'status': 'failed', 'error': str(e)}
            self.arima_fitted = None
        
        self.is_fitted = True
        return training_results
    
    def _ensure_prophet_fitted(self):
        """Ensure Prophet model is fitted, refit if necessary"""
        # Check if we have a model at all
        if self.prophet_model is None:
            logger.error(f"No Prophet model exists for {self.metric_name}")
            return False
        
        # Simple check: try to access the params attribute
        try:
            params = getattr(self.prophet_model, 'params', None)
            if params is not None and len(params) > 0:
                logger.debug(f"Prophet model appears fitted for {self.metric_name}")
                return True
        except Exception as e:
            logger.warning(f"Error checking Prophet model state for {self.metric_name}: {e}")
        
        # Model not fitted, try to refit if we have training data
        if not hasattr(self, 'prophet_training_data') or self.prophet_training_data.empty:
            logger.error(f"Cannot refit Prophet model for {self.metric_name} - no training data available")
            return False
        
        try:
            logger.info(f"Attempting to refit Prophet model for {self.metric_name}")
            
            # Get the stored configuration
            config = getattr(self, 'prophet_config_used', {})
            
            # Create a fresh Prophet model
            self.prophet_model = Prophet(
                growth=config.get('growth', 'linear'),
                daily_seasonality=config.get('daily_seasonality', True),
                weekly_seasonality=config.get('weekly_seasonality', True),
                yearly_seasonality=config.get('yearly_seasonality', False),
                changepoint_prior_scale=config.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=config.get('seasonality_prior_scale', 10.0),
                interval_width=config.get('interval_width', 0.80)
            )
            
            # Add seasonalities if needed
            for pattern in self.profile.seasonal_patterns:
                if pattern == 'business_hours':
                    self.prophet_model.add_seasonality(
                        name='business_hours',
                        period=1,
                        fourier_order=3,
                        condition_name='is_business_hour'
                    )
            
            # Refit with stored data
            training_data = self.prophet_training_data.copy()
            logger.info(f"Refitting with {len(training_data)} samples")
            
            self.prophet_model.fit(training_data)
            
            # Verify refit worked
            if hasattr(self.prophet_model, 'params') and self.prophet_model.params is not None:
                logger.info(f"Successfully refitted Prophet model for {self.metric_name}")
                return True
            else:
                logger.error(f"Refit failed for {self.metric_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to refit Prophet model for {self.metric_name}: {e}")
            return False
    
    def predict(self, periods: int = 24, frequency: str = 'H') -> Dict[str, pd.DataFrame]:
        """Generate predictions from both models with improved error handling"""
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        logger.info(f"Starting prediction for {self.metric_name} - periods: {periods}")
        predictions = {}
        
        # Prophet predictions
        if self.prophet_model is not None:
            try:
                logger.info(f"Attempting Prophet prediction for {self.metric_name}")
                
                # Ensure model is fitted
                if not self._ensure_prophet_fitted():
                    logger.error(f"Cannot ensure Prophet model is fitted for {self.metric_name}")
                else:
                    # Create future dataframe
                    future = self.prophet_model.make_future_dataframe(periods=periods, freq=frequency)
                    logger.info(f"Created future dataframe with {len(future)} rows")
                    
                    # Add required columns based on stored training data
                    if hasattr(self, 'prophet_training_data'):
                        training_columns = set(self.prophet_training_data.columns)
                        
                        # Add business hour indicator if it was in training
                        if 'is_business_hour' in training_columns:
                            future['is_business_hour'] = (
                                (future['ds'].dt.hour >= 9) & 
                                (future['ds'].dt.hour <= 17) & 
                                (future['ds'].dt.dayofweek < 5)
                            )
                            logger.debug("Added is_business_hour column to future dataframe")
                        
                        # Add capacity constraints if they were in training
                        if 'cap' in training_columns:
                            cap_value = self.profile.prophet_config.get('cap', 100.0)
                            future['cap'] = cap_value
                            logger.debug(f"Added cap={cap_value} to future dataframe")
                        
                        if 'floor' in training_columns:
                            floor_value = self.profile.prophet_config.get('floor', 0.0)
                            future['floor'] = floor_value
                            logger.debug(f"Added floor={floor_value} to future dataframe")
                    
                    # Generate forecast
                    logger.info(f"Generating Prophet forecast for {self.metric_name}...")
                    forecast = self.prophet_model.predict(future)
                    logger.info(f"Prophet forecast generated successfully")
                    
                    # Extract future predictions only
                    future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
                    
                    # Apply metric-specific constraints
                    if self.profile.data_type == 'percentage':
                        future_forecast['yhat'] = future_forecast['yhat'].clip(0, 100)
                        future_forecast['yhat_lower'] = future_forecast['yhat_lower'].clip(0, 100)
                        future_forecast['yhat_upper'] = future_forecast['yhat_upper'].clip(0, 100)
                    
                    predictions['prophet'] = future_forecast
                    logger.info(f"Prophet predictions ready: {len(future_forecast)} points")
                    
            except Exception as e:
                logger.error(f"Prophet prediction failed for {self.metric_name}: {e}")
                logger.exception("Full Prophet prediction error:")
        
        # ARIMA predictions with improved date handling
        if hasattr(self, 'arima_fitted') and self.arima_fitted is not None:
            try:
                logger.info(f"Attempting ARIMA prediction for {self.metric_name}")
                
                arima_forecast = self.arima_fitted.forecast(steps=periods)
                
                # Create future dates - use a more reliable method
                # Try to get the last date from training data first
                if hasattr(self, 'prophet_training_data') and not self.prophet_training_data.empty:
                    last_timestamp = self.prophet_training_data['ds'].max()
                    logger.info(f"Using last training timestamp: {last_timestamp}")
                else:
                    # Fallback to current time
                    last_timestamp = pd.Timestamp.now().floor('H')
                    logger.info(f"Using current timestamp as fallback: {last_timestamp}")
                
                # Remove timezone if present
                if hasattr(last_timestamp, 'tz') and last_timestamp.tz is not None:
                    last_timestamp = last_timestamp.tz_localize(None)
                
                future_dates = pd.date_range(
                    start=last_timestamp + pd.Timedelta(hours=1),
                    periods=periods,
                    freq=frequency
                )
                
                # Handle forecast format (could be Series or array)
                if hasattr(arima_forecast, 'values'):
                    forecast_values = arima_forecast.values
                else:
                    forecast_values = np.array(arima_forecast)
                
                # Get confidence intervals
                try:
                    conf_int = self.arima_fitted.get_forecast(steps=periods).conf_int()
                    lower_bounds = conf_int.iloc[:, 0].values
                    upper_bounds = conf_int.iloc[:, 1].values
                except:
                    # Fallback confidence intervals
                    lower_bounds = forecast_values * 0.9
                    upper_bounds = forecast_values * 1.1
                
                predictions['arima'] = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': forecast_values,
                    'yhat_lower': lower_bounds,
                    'yhat_upper': upper_bounds
                })
                
                logger.info(f"ARIMA predictions ready: {len(predictions['arima'])} points")
                
            except Exception as e:
                logger.error(f"ARIMA prediction failed for {self.metric_name}: {e}")
        
        if not predictions:
            logger.warning(f"No predictions generated for {self.metric_name}")
        else:
            logger.info(f"Prediction complete for {self.metric_name}. Models: {list(predictions.keys())}")
        
        return predictions
    
    def _preprocess_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Apply metric-specific preprocessing"""
        processed_df = df.copy()
        config = self.profile.preprocessing_config
        
        # Handle outliers based on metric type
        if config.get('outlier_method') == 'iqr':
            q1 = processed_df[target_column].quantile(0.25)
            q3 = processed_df[target_column].quantile(0.75)
            iqr = q3 - q1
            factor = config.get('outlier_factor', 1.5)
            
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            
            mask = (processed_df[target_column] >= lower_bound) & (processed_df[target_column] <= upper_bound)
            processed_df = processed_df[mask]
            
        elif config.get('outlier_method') == 'percentile':
            percentile = config.get('outlier_percentile', 95)
            threshold = processed_df[target_column].quantile(percentile / 100.0)
            processed_df = processed_df[processed_df[target_column] <= threshold]
        
        # Apply scaling if needed
        if 'scale_factor' in config:
            scale_factor = config['scale_factor']
            if scale_factor != 1:
                processed_df[target_column] = processed_df[target_column] / scale_factor
                logger.info(f"Applied scaling factor {scale_factor} to {target_column}")
        
        # Smooth spikes if requested
        if config.get('smooth_spikes', False):
            processed_df[target_column] = processed_df[target_column].rolling(
                window=3, center=True, min_periods=1
            ).median()
        
        # Handle missing values
        fill_method = config.get('fill_method', 'interpolate')
        if fill_method == 'interpolate':
            processed_df[target_column] = processed_df[target_column].interpolate(method='time')
        elif fill_method == 'forward_fill':
            processed_df[target_column] = processed_df[target_column].fillna(method='ffill')
        
        return processed_df
    
    def _calculate_data_quality(self, df: pd.DataFrame, target_column: str) -> float:
        """Calculate data quality score (0-1)"""
        if df.empty:
            return 0.0
        
        data = df[target_column]
        
        # Components of data quality
        completeness = 1.0 - (data.isna().sum() / len(data))
        
        # Consistency (low variance in differences indicates consistent collection)
        if len(data) > 1:
            consistency = 1.0 / (1.0 + data.diff().std())
        else:
            consistency = 0.5
        
        # Reasonable range (penalize extreme outliers)
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((data < (q1 - 3 * iqr)) | (data > (q3 + 3 * iqr))).sum()
        range_quality = 1.0 - (outliers / len(data))
        
        # Overall score (weighted average)
        quality_score = (0.4 * completeness + 0.3 * consistency + 0.3 * range_quality)
        return min(1.0, max(0.0, quality_score))
    
    def _prepare_prophet_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Prepare data for Prophet with enhanced validation"""
        prophet_data = df.reset_index()
        prophet_data = prophet_data.rename(columns={
            prophet_data.columns[0]: 'ds',
            target_column: 'y'
        })
        
        # Ensure datetime index and REMOVE timezone
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # CRITICAL FIX: Remove timezone information for Prophet compatibility
        if prophet_data['ds'].dt.tz is not None:
            prophet_data['ds'] = prophet_data['ds'].dt.tz_localize(None)
            logger.info(f"Removed timezone from datetime column for Prophet compatibility")
        
        # Remove duplicates and sort
        prophet_data = prophet_data.drop_duplicates(subset=['ds']).sort_values('ds')
        
        # Remove any remaining NaN values
        prophet_data = prophet_data.dropna(subset=['y'])
        
        # Also clean up extra columns that Prophet doesn't need
        # Keep only ds, y, and any additional columns we explicitly added
        required_columns = ['ds', 'y']
        
        # Add capacity constraints if they exist
        if 'cap' in prophet_data.columns:
            required_columns.append('cap')
        if 'floor' in prophet_data.columns:
            required_columns.append('floor')
        if 'is_business_hour' in prophet_data.columns:
            required_columns.append('is_business_hour')
        
        # Filter to only required columns
        prophet_data = prophet_data[required_columns]
        
        logger.info(f"Prepared Prophet data with {len(prophet_data)} samples, columns: {list(prophet_data.columns)}")
        
        return prophet_data
    
    def _find_optimal_arima_order(self, ts_data: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA order for specific metric type"""
        
        # Metric-specific parameter ranges
        if self.profile.data_type == 'percentage':
            p_range = range(0, 3)
            d_range = range(0, 2)
            q_range = range(0, 3)
        elif self.profile.data_type == 'load_average':
            p_range = range(0, 4)
            d_range = range(0, 2)
            q_range = range(0, 3)
        elif self.profile.data_type == 'process_count':
            p_range = range(0, 2)
            d_range = range(0, 1)
            q_range = range(0, 2)
        else:
            p_range = range(0, 3)
            d_range = range(0, 2)
            q_range = range(0, 3)
        
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        from statsmodels.tsa.arima.model import ARIMA
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order

class PredictionEngineManager:
    """Enhanced prediction engine manager"""
    
    def __init__(self):
        self.forecasters = {}
        self.analyzer = MetricAnalyzer()
        self.models_dir = settings.ai.ml_storage_path / "prediction_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def train_prediction_models(
        self, 
        metrics_df: pd.DataFrame,
        target_metrics: List[str] = None,
        forecast_horizon: int = 24
    ) -> Dict[str, PredictionMetadata]:
        """Train prediction models with enhanced metric-specific optimization"""
        
        if metrics_df.empty:
            raise ValueError("No data provided for training")
        
        # Use all available metrics if none specified
        if target_metrics is None:
            target_metrics = [col for col in metrics_df.columns if col in settings.monitoring.key_metrics]
        
        trained_models = {}
        
        for metric in target_metrics:
            if metric not in metrics_df.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue
            
            try:
                logger.info(f"Training enhanced forecasting model for {metric}")
                
                # Analyze metric characteristics
                profile = self.analyzer.analyze_metric_characteristics(metrics_df, metric)
                
                # Create enhanced forecaster
                forecaster = EnhancedTimeSeriesForecaster(metric, profile)
                
                # Split data for validation
                split_point = int(len(metrics_df) * 0.8)
                train_data = metrics_df.iloc[:split_point]
                test_data = metrics_df.iloc[split_point:]
                
                # Train the model
                training_results = forecaster.fit(train_data, metric)
                
                # Evaluate accuracy
                accuracy_metrics = forecaster.get_forecast_accuracy(test_data, metric)
                
                # Store forecaster
                self.forecasters[metric] = forecaster
                
                # Create metadata
                metadata = PredictionMetadata(
                    model_name=f"enhanced_forecaster_{metric}",
                    model_type="enhanced_time_series",
                    version=f"1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    training_date=datetime.now(),
                    training_samples=len(train_data),
                    features_count=1,
                    prediction_horizon=f"{forecast_horizon}h",
                    performance_metrics=clean_for_json(accuracy_metrics),
                    hyperparameters=clean_for_json(profile.prophet_config),
                    model_path=str(self.models_dir / f"{metric}_model.joblib")
                )
                
                trained_models[metric] = metadata
                
                logger.info(f"Successfully trained model for {metric} with data quality score: {forecaster.data_quality_score:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train model for {metric}: {e}")
                continue
        
        # Save all models
        self._save_models()
        
        return trained_models
    
    def generate_forecasts(
        self,
        forecast_horizon: int = 24,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Generate enhanced forecasts with confidence intervals and insights"""
        
        if metrics is None:
            metrics = list(self.forecasters.keys())
        
        forecasts = {}
        forecast_summary = {
            'total_metrics': 0,
            'successful_forecasts': 0,
            'average_confidence': 0.0,
            'horizon_hours': forecast_horizon,
            'generated_at': datetime.now().isoformat()
        }
        
        confidence_scores = []
        
        for metric in metrics:
            if metric not in self.forecasters:
                logger.warning(f"No trained forecaster for {metric}")
                continue
            
            try:
                forecaster = self.forecasters[metric]
                predictions = forecaster.predict(periods=forecast_horizon, frequency='H')
                
                # Process and enhance predictions
                metric_forecast = {
                    'profile': {
                        'data_type': forecaster.profile.data_type,
                        'unit': forecaster.profile.unit,
                        'quality_score': forecaster.data_quality_score
                    },
                    'models': {}
                }
                
                for model_name, pred_df in predictions.items():
                    if not pred_df.empty:
                        # Calculate prediction confidence
                        pred_values = pred_df['yhat'].values
                        lower_bounds = pred_df['yhat_lower'].values
                        upper_bounds = pred_df['yhat_upper'].values
                        
                        # Confidence based on interval width
                        interval_widths = upper_bounds - lower_bounds
                        avg_interval = np.mean(interval_widths)
                        avg_value = np.mean(pred_values)
                        confidence = max(0, 1 - (avg_interval / (avg_value + 1e-6)))
                        confidence_scores.append(confidence)
                        
                        metric_forecast['models'][model_name] = {
                            'timestamps': pred_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                            'predicted_values': pred_values.tolist(),
                            'lower_bound': lower_bounds.tolist(),
                            'upper_bound': upper_bounds.tolist(),
                            'confidence_score': confidence,
                            'trend_direction': self._calculate_trend(pred_values),
                            'anomaly_alerts': self._detect_forecast_anomalies(
                                pred_values, forecaster.profile
                            )
                        }
                
                forecasts[metric] = metric_forecast
                forecast_summary['successful_forecasts'] += 1
                
            except Exception as e:
                logger.error(f"Failed to generate forecast for {metric}: {e}")
                forecasts[metric] = {'error': str(e)}
            
            forecast_summary['total_metrics'] += 1
        
        # Calculate average confidence
        if confidence_scores:
            forecast_summary['average_confidence'] = np.mean(confidence_scores)
        
        return {
            'forecasts': forecasts,
            'summary': forecast_summary
        }
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction from predicted values"""
        if len(values) < 2:
            return 'stable'
        
        # Linear regression on values
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _detect_forecast_anomalies(self, values: np.ndarray, profile: MetricProfile) -> List[Dict[str, Any]]:
        """Detect potential anomalies in forecast"""
        anomalies = []
        
        # Check against metric-specific thresholds
        if profile.data_type == 'percentage':
            # Check for values approaching limits
            for i, val in enumerate(values):
                if val > 90:
                    anomalies.append({
                        'hour': i + 1,
                        'type': 'high_utilization',
                        'value': val,
                        'severity': 'warning' if val < 95 else 'critical'
                    })
                elif val < 5 and profile.name not in ['swap_size_pfree']:
                    anomalies.append({
                        'hour': i + 1,
                        'type': 'unusually_low',
                        'value': val,
                        'severity': 'info'
                    })
        
        elif profile.data_type == 'load_average':
            # Check for high load
            for i, val in enumerate(values):
                if val > 4.0:  # Assuming quad-core system
                    anomalies.append({
                        'hour': i + 1,
                        'type': 'high_load',
                        'value': val,
                        'severity': 'warning' if val < 8 else 'critical'
                    })
        
        return anomalies
    
    def get_prediction_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all prediction models"""
        status = {
            'total_models': len(self.forecasters),
            'models': {},
            'system_health': {
                'average_data_quality': 0.0,
                'model_coverage': 0.0,
                'last_update': None
            }
        }
        
        quality_scores = []
        
        for metric, forecaster in self.forecasters.items():
            model_status = {
                'is_trained': forecaster.is_fitted,
                'data_type': forecaster.profile.data_type,
                'unit': forecaster.profile.unit,
                'data_quality_score': forecaster.data_quality_score,
                'seasonal_patterns': forecaster.profile.seasonal_patterns,
                'accuracy_targets': forecaster.profile.accuracy_targets
            }
            
            status['models'][metric] = model_status
            quality_scores.append(forecaster.data_quality_score)
        
        # Calculate system health metrics
        if quality_scores:
            status['system_health']['average_data_quality'] = np.mean(quality_scores)
        
        total_key_metrics = len(settings.monitoring.key_metrics)
        status['system_health']['model_coverage'] = len(self.forecasters) / total_key_metrics if total_key_metrics > 0 else 0
        
        return status
    
    def load_predictors(self, metrics: List[str] = None) -> bool:
        """Load saved prediction models with enhanced state restoration"""
        try:
            import joblib
            
            forecasters_path = self.models_dir / "enhanced_forecasters.joblib"
            if not forecasters_path.exists():
                logger.warning("No saved forecasters found")
                return False
            
            saved_data = joblib.load(forecasters_path)
            
            loaded_count = 0
            for metric_name, forecaster_data in saved_data.get('forecasters', {}).items():
                if metrics is None or metric_name in metrics:
                    try:
                        # Restore the forecaster
                        forecaster = forecaster_data['forecaster']
                        
                        # Restore additional state if available
                        if 'training_data' in forecaster_data and forecaster_data['training_data'] is not None:
                            forecaster.prophet_training_data = forecaster_data['training_data']
                        
                        if 'config_used' in forecaster_data:
                            forecaster.prophet_config_used = forecaster_data['config_used']
                        
                        if 'data_quality_score' in forecaster_data:
                            forecaster.data_quality_score = forecaster_data['data_quality_score']
                        
                        self.forecasters[metric_name] = forecaster
                        loaded_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to load forecaster for {metric_name}: {e}")
            
            # Restore profiles
            self.analyzer.metric_profiles = saved_data.get('profiles', {})
            
            logger.info(f"Loaded {loaded_count} enhanced forecasters")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Failed to load prediction models: {e}")
            return False
    
    def _save_models(self):
        """Save all trained models and profiles with enhanced state preservation"""
        try:
            import joblib
            
            # Prepare data for saving with enhanced state preservation
            save_data = {
                'forecasters': {},
                'profiles': self.analyzer.metric_profiles,
                'saved_at': datetime.now().isoformat()
            }
            
            # Save each forecaster with its complete state
            for metric_name, forecaster in self.forecasters.items():
                forecaster_data = {
                    'forecaster': forecaster,
                    'profile': forecaster.profile,
                    'training_data': getattr(forecaster, 'prophet_training_data', None),
                    'config_used': getattr(forecaster, 'prophet_config_used', {}),
                    'data_quality_score': getattr(forecaster, 'data_quality_score', 0.0)
                }
                save_data['forecasters'][metric_name] = forecaster_data
            
            models_path = self.models_dir / "enhanced_forecasters.joblib"
            joblib.dump(save_data, models_path)
            logger.info(f"Enhanced prediction models saved: {models_path}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

# Global prediction manager instance
prediction_manager = PredictionEngineManager()

