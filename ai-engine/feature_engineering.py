import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

from config import settings

logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """Centralized feature engineering pipeline for consistency across all models"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.feature_names = []
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the pipeline and transform data"""
        if df.empty:
            logger.warning("Empty DataFrame provided to feature engineering")
            return df
            
        # Store original feature names for consistency
        self.original_features = list(df.columns)
        
        # Clean data first
        df_clean = self._clean_data(df.copy())
        
        # Create temporal features
        df_temporal = self._create_temporal_features(df_clean)
        
        # Create statistical features
        df_statistical = self._create_statistical_features(df_temporal)
        
        # Create domain-specific features
        df_domain = self._create_domain_features(df_statistical)
        
        # Scale features
        df_scaled = self._scale_features(df_domain, fit=True)
        
        # Store final feature names
        self.feature_names = list(df_scaled.columns)
        self.is_fitted = True
        
        logger.info(f"Feature engineering completed: {len(self.feature_names)} features created")
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
            
        if df.empty:
            logger.warning("Empty DataFrame provided for transformation")
            return pd.DataFrame(columns=self.feature_names)
        
        # Ensure same features as training
        df_aligned = self._align_features(df.copy())
        
        # Apply same transformations
        df_clean = self._clean_data(df_aligned)
        df_temporal = self._create_temporal_features(df_clean)
        df_statistical = self._create_statistical_features(df_temporal)
        df_domain = self._create_domain_features(df_statistical)
        df_scaled = self._scale_features(df_domain, fit=False)
        
        # Ensure output has same feature order
        df_final = df_scaled.reindex(columns=self.feature_names, fill_value=0.0)
        
        return df_final
    
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure input features match training features"""
        # Add missing features
        for feature in self.original_features:
            if feature not in df.columns:
                df[feature] = 0.0
                logger.warning(f"Missing feature '{feature}' filled with zeros")
        
        # Remove extra features
        extra_features = [col for col in df.columns if col not in self.original_features]
        if extra_features:
            df = df.drop(columns=extra_features)
            logger.warning(f"Removed {len(extra_features)} extra features")
        
        # Reorder to match training
        df = df.reindex(columns=self.original_features, fill_value=0.0)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and impute missing data"""
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Impute missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Use different imputation strategies
            if len(df) > 50:  # Use KNN for larger datasets
                if 'knn_imputer' not in self.imputers:
                    self.imputers['knn_imputer'] = KNNImputer(n_neighbors=5)
                    imputed_data = self.imputers['knn_imputer'].fit_transform(df[numeric_cols])
                else:
                    imputed_data = self.imputers['knn_imputer'].transform(df[numeric_cols])
            else:  # Use median for smaller datasets
                if 'median_imputer' not in self.imputers:
                    self.imputers['median_imputer'] = SimpleImputer(strategy='median')
                    imputed_data = self.imputers['median_imputer'].fit_transform(df[numeric_cols])
                else:
                    imputed_data = self.imputers['median_imputer'].transform(df[numeric_cols])
            
            df[numeric_cols] = imputed_data
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df_temporal = df.copy()
        
        # Extract time information from index or timestamp column
        if isinstance(df.index, pd.DatetimeIndex):
            time_series = df.index
        elif 'timestamp' in df.columns:
            time_series = pd.to_datetime(df['timestamp'])
        else:
            # Create dummy temporal features
            logger.warning("No timestamp found, creating dummy temporal features")
            current_time = datetime.now()
            time_series = pd.date_range(start=current_time, periods=len(df), freq='5T')
        
        # Time-based features
        df_temporal['hour'] = time_series.hour
        df_temporal['day_of_week'] = time_series.dayofweek
        df_temporal['is_weekend'] = (time_series.dayofweek >= 5).astype(int)
        df_temporal['is_business_hours'] = (
            (time_series.hour >= 9) & 
            (time_series.hour <= 17) & 
            (time_series.dayofweek < 5)
        ).astype(int)
        df_temporal['hour_sin'] = np.sin(2 * np.pi * time_series.hour / 24)
        df_temporal['hour_cos'] = np.cos(2 * np.pi * time_series.hour / 24)
        df_temporal['day_sin'] = np.sin(2 * np.pi * time_series.dayofweek / 7)
        df_temporal['day_cos'] = np.cos(2 * np.pi * time_series.dayofweek / 7)
        
        return df_temporal
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from raw metrics"""
        df_stat = df.copy()
        numeric_cols = df_stat.select_dtypes(include=[np.number]).columns
        
        # Remove temporal features from statistical calculations
        metric_cols = [col for col in numeric_cols if not any(
            temporal in col.lower() for temporal in ['hour', 'day', 'weekend', 'business', 'sin', 'cos']
        )]
        
        for col in metric_cols:
            if df_stat[col].nunique() > 1:  # Only if there's variation
                # Rolling statistics
                for window in [3, 5, 10]:
                    df_stat[f'{col}_rolling_mean_{window}'] = df_stat[col].rolling(window, min_periods=1).mean()
                    df_stat[f'{col}_rolling_std_{window}'] = df_stat[col].rolling(window, min_periods=1).std().fillna(0)
                    df_stat[f'{col}_rolling_max_{window}'] = df_stat[col].rolling(window, min_periods=1).max()
                    df_stat[f'{col}_rolling_min_{window}'] = df_stat[col].rolling(window, min_periods=1).min()
                
                # Lag features
                for lag in [1, 2, 3]:
                    df_stat[f'{col}_lag_{lag}'] = df_stat[col].shift(lag).fillna(df_stat[col].iloc[0])
                
                # Rate of change
                df_stat[f'{col}_pct_change'] = df_stat[col].pct_change().fillna(0)
                df_stat[f'{col}_diff'] = df_stat[col].diff().fillna(0)
                
                # Z-score (standardization within each metric)
                if df_stat[col].std() > 0:
                    df_stat[f'{col}_zscore'] = (df_stat[col] - df_stat[col].mean()) / df_stat[col].std()
                else:
                    df_stat[f'{col}_zscore'] = 0
        
        return df_stat
    
    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create monitoring-specific domain features"""
        df_domain = df.copy()
        
        # CPU-related features
        cpu_cols = [col for col in df.columns if 'cpu' in col.lower() and 'rolling' not in col]
        if cpu_cols:
            df_domain['cpu_avg'] = df_domain[cpu_cols].mean(axis=1)
            df_domain['cpu_max'] = df_domain[cpu_cols].max(axis=1)
            df_domain['cpu_utilization_severity'] = np.where(
                df_domain['cpu_avg'] > 90, 3,  # Critical
                np.where(df_domain['cpu_avg'] > 80, 2,  # High
                np.where(df_domain['cpu_avg'] > 70, 1, 0))  # Normal
            )
        
        # Memory-related features
        memory_cols = [col for col in df.columns if any(mem in col.lower() for mem in ['memory', 'ram']) and 'rolling' not in col]
        if memory_cols:
            df_domain['memory_avg'] = df_domain[memory_cols].mean(axis=1)
            df_domain['memory_pressure'] = np.where(df_domain['memory_avg'] > 95, 1, 0)
        
        # Disk-related features
        disk_cols = [col for col in df.columns if any(disk in col.lower() for disk in ['disk', 'fs', 'filesystem']) and 'rolling' not in col]
        if disk_cols:
            df_domain['disk_avg'] = df_domain[disk_cols].mean(axis=1)
            df_domain['disk_critical'] = np.where(df_domain['disk_avg'] > 90, 1, 0)
        
        # Network-related features
        network_cols = [col for col in df.columns if 'net' in col.lower() and 'rolling' not in col]
        if network_cols:
            df_domain['network_avg'] = df_domain[network_cols].mean(axis=1)
        
        # System health score
        health_components = []
        if 'cpu_avg' in df_domain.columns:
            health_components.append((100 - df_domain['cpu_avg']) / 100)
        if 'memory_avg' in df_domain.columns:
            health_components.append((100 - df_domain['memory_avg']) / 100)
        if 'disk_avg' in df_domain.columns:
            health_components.append((100 - df_domain['disk_avg']) / 100)
        
        if health_components:
            df_domain['system_health_score'] = np.mean(health_components, axis=0)
        else:
            df_domain['system_health_score'] = 1.0
        
        return df_domain
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale features using robust scaling"""
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df_scaled
        
        if fit:
            self.scalers['robust'] = RobustScaler()
            scaled_data = self.scalers['robust'].fit_transform(df_scaled[numeric_cols])
        else:
            if 'robust' not in self.scalers:
                logger.warning("Scaler not fitted, using identity transformation")
                return df_scaled
            scaled_data = self.scalers['robust'].transform(df_scaled[numeric_cols])
        
        df_scaled[numeric_cols] = scaled_data
        
        return df_scaled
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by category for analysis"""
        if not self.is_fitted:
            return {}
        
        groups = {
            'temporal': [f for f in self.feature_names if any(x in f.lower() for x in ['hour', 'day', 'weekend', 'business', 'sin', 'cos'])],
            'cpu': [f for f in self.feature_names if 'cpu' in f.lower()],
            'memory': [f for f in self.feature_names if any(x in f.lower() for x in ['memory', 'ram'])],
            'disk': [f for f in self.feature_names if any(x in f.lower() for x in ['disk', 'fs', 'filesystem'])],
            'network': [f for f in self.feature_names if 'net' in f.lower()],
            'rolling': [f for f in self.feature_names if 'rolling' in f.lower()],
            'lag': [f for f in self.feature_names if 'lag' in f.lower()],
            'derived': [f for f in self.feature_names if any(x in f.lower() for x in ['avg', 'max', 'severity', 'pressure', 'critical', 'health'])]
        }
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups

# Global feature engineering pipeline
feature_pipeline = FeatureEngineeringPipeline()

