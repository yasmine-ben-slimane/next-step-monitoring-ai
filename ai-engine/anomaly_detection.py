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

# ML libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import scipy.stats as stats

# Optional: SHAP for explainable AI (add to requirements.txt if needed)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

def clean_float_values(data: Any) -> Any:
    """Clean float values to ensure JSON compatibility"""
    if isinstance(data, dict):
        return {k: clean_float_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_float_values(item) for item in data]
    elif isinstance(data, (int, float)):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return float(data)
    return data

@dataclass
class ModelMetadata:
    """Metadata for trained anomaly detection models"""
    model_name: str
    version: str
    training_date: datetime
    training_samples: int
    features_count: int
    performance_metrics: Dict[str, float]
    contamination_rate: float
    hyperparameters: Dict[str, Any]
    data_characteristics: Dict[str, Any]
    model_path: str

class StatisticalAnomalyDetector:
    """Statistical methods for anomaly detection"""
    
    def __init__(self, z_threshold: float = 3.0, modified_z_threshold: float = 3.5):
        self.z_threshold = z_threshold
        self.modified_z_threshold = modified_z_threshold
        self.baseline_stats = {}
        
    def fit(self, X: pd.DataFrame):
        """Calculate baseline statistics for each feature"""
        self.baseline_stats = {}
        
        for column in X.select_dtypes(include=[np.number]).columns:
            data = X[column].dropna()
            
            if len(data) == 0:
                continue
                
            mean_val = data.mean()
            std_val = data.std()
            median_val = data.median()
            mad_val = stats.median_abs_deviation(data)
            q1_val = data.quantile(0.25)
            q3_val = data.quantile(0.75)
            iqr_val = q3_val - q1_val
            
            # Handle edge cases
            if pd.isna(std_val) or std_val == 0:
                std_val = 1e-8  # Small value to prevent division by zero
            if pd.isna(mad_val) or mad_val == 0:
                mad_val = 1e-8
            if pd.isna(iqr_val) or iqr_val == 0:
                iqr_val = 1e-8
            
            self.baseline_stats[column] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'median': float(median_val),
                'mad': float(mad_val),
                'q1': float(q1_val),
                'q3': float(q3_val),
                'iqr': float(iqr_val)
            }
        
        logger.info(f"Statistical baseline calculated for {len(self.baseline_stats)} features")
        return self
    
    def detect_z_score_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """Detect anomalies using Z-score method"""
        if not self.baseline_stats:
            raise ValueError("Model not fitted. Call fit() first.")
            
        anomalies = np.zeros(len(X))
        
        for column in X.select_dtypes(include=[np.number]).columns:
            if column in self.baseline_stats:
                stats_data = self.baseline_stats[column]
                z_scores = np.abs((X[column] - stats_data['mean']) / stats_data['std'])
                
                # Handle NaN/inf values
                z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
                anomalies += (z_scores > self.z_threshold).astype(int)
        
        return (anomalies > 0).astype(int)
    
    def detect_modified_z_score_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """Detect anomalies using Modified Z-score (more robust)"""
        if not self.baseline_stats:
            raise ValueError("Model not fitted. Call fit() first.")
            
        anomalies = np.zeros(len(X))
        
        for column in X.select_dtypes(include=[np.number]).columns:
            if column in self.baseline_stats:
                stats_data = self.baseline_stats[column]
                modified_z_scores = 0.6745 * (X[column] - stats_data['median']) / stats_data['mad']
                
                # Handle NaN/inf values
                modified_z_scores = np.nan_to_num(modified_z_scores, nan=0.0, posinf=0.0, neginf=0.0)
                anomalies += (np.abs(modified_z_scores) > self.modified_z_threshold).astype(int)
        
        return (anomalies > 0).astype(int)
    
    def detect_iqr_anomalies(self, X: pd.DataFrame, multiplier: float = 1.5) -> np.ndarray:
        """Detect anomalies using Interquartile Range method"""
        if not self.baseline_stats:
            raise ValueError("Model not fitted. Call fit() first.")
            
        anomalies = np.zeros(len(X))
        
        for column in X.select_dtypes(include=[np.number]).columns:
            if column in self.baseline_stats:
                stats_data = self.baseline_stats[column]
                lower_bound = stats_data['q1'] - multiplier * stats_data['iqr']
                upper_bound = stats_data['q3'] + multiplier * stats_data['iqr']
                
                column_anomalies = (X[column] < lower_bound) | (X[column] > upper_bound)
                anomalies += column_anomalies.astype(int)
        
        return (anomalies > 0).astype(int)
    
    def get_anomaly_scores(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get detailed anomaly scores from all statistical methods"""
        return {
            'z_score': self.detect_z_score_anomalies(X),
            'modified_z_score': self.detect_modified_z_score_anomalies(X),
            'iqr': self.detect_iqr_anomalies(X)
        }

class EnsembleAnomalyDetector:
    """Ensemble anomaly detection combining multiple algorithms"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models = {}
        self.statistical_detector = StatisticalAnomalyDetector()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all anomaly detection models"""
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                n_jobs=-1
            ),
            'lof': LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20,
                novelty=True,
                n_jobs=-1
            ),
            'statistical': self.statistical_detector
        }
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EnsembleAnomalyDetector':
        """Fit all anomaly detection models"""
        logger.info(f"Training ensemble anomaly detector on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Prepare data
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Clean data before scaling
        X_numeric = X_numeric.fillna(0)  # Fill NaN with 0
        X_numeric = X_numeric.replace([np.inf, -np.inf], 0)  # Replace inf with 0
        
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Fit models
        self.models['isolation_forest'].fit(X_scaled)
        self.models['lof'].fit(X_scaled)
        self.models['statistical'].fit(X_numeric)
        
        self.is_fitted = True
        logger.info("Ensemble anomaly detector trained successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies using ensemble voting"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Clean data before scaling
        X_numeric = X_numeric.fillna(0)
        X_numeric = X_numeric.replace([np.inf, -np.inf], 0)
        
        X_scaled = self.scaler.transform(X_numeric)
        
        # Get predictions from each model
        predictions = {}
        
        # Isolation Forest (-1 for anomaly, 1 for normal)
        if_pred = self.models['isolation_forest'].predict(X_scaled)
        predictions['isolation_forest'] = (if_pred == -1).astype(int)
        
        # Local Outlier Factor (-1 for anomaly, 1 for normal)
        lof_pred = self.models['lof'].predict(X_scaled)
        predictions['lof'] = (lof_pred == -1).astype(int)
        
        # Statistical methods (majority vote)
        stat_scores = self.models['statistical'].get_anomaly_scores(X_numeric)
        stat_votes = np.array([stat_scores[method] for method in stat_scores.keys()])
        predictions['statistical'] = (np.sum(stat_votes, axis=0) >= 2).astype(int)
        
        # Ensemble voting (majority wins)
        all_predictions = np.array([predictions[model] for model in predictions.keys()])
        ensemble_prediction = (np.sum(all_predictions, axis=0) >= 2).astype(int)
        
        return ensemble_prediction
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly confidence scores (0-1 range)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Clean data before scaling
        X_numeric = X_numeric.fillna(0)
        X_numeric = X_numeric.replace([np.inf, -np.inf], 0)
        
        X_scaled = self.scaler.transform(X_numeric)
        
        # Get anomaly scores
        if_scores = self.models['isolation_forest'].decision_function(X_scaled)
        lof_scores = self.models['lof'].decision_function(X_scaled)
        
        # Clean scores
        if_scores = np.nan_to_num(if_scores, nan=0.0, posinf=1.0, neginf=-1.0)
        lof_scores = np.nan_to_num(lof_scores, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize scores to 0-1 range (higher = more anomalous)
        if len(np.unique(if_scores)) > 1:
            if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        else:
            if_scores_norm = np.zeros_like(if_scores)
            
        if len(np.unique(lof_scores)) > 1:
            lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
        else:
            lof_scores_norm = np.zeros_like(lof_scores)
        
        # Invert scores (higher values = more anomalous)
        if_scores_norm = 1 - if_scores_norm
        lof_scores_norm = 1 - lof_scores_norm
        
        # Average ensemble scores
        ensemble_scores = (if_scores_norm + lof_scores_norm) / 2
        
        # Final cleaning
        ensemble_scores = np.nan_to_num(ensemble_scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        return ensemble_scores
    
    def explain_anomalies(self, X: pd.DataFrame, anomaly_indices: list[int]) -> Dict[int, Dict[str, float]]:
        """Explain which features contributed to anomaly detection"""
        explanations = {}
        
        X_numeric = X.select_dtypes(include=[np.number])
        
        for idx in anomaly_indices:
            if idx < len(X_numeric):
                row_explanation = {}
                
                # Statistical feature contributions
                for column in X_numeric.columns:
                    if column in self.statistical_detector.baseline_stats:
                        stats_data = self.statistical_detector.baseline_stats[column]
                        value = X_numeric.iloc[idx][column]
                        
                        # Handle missing/invalid values
                        if pd.isna(value) or np.isinf(value):
                            value = 0.0
                        
                        # Calculate deviation from normal
                        z_score = abs((value - stats_data['mean']) / stats_data['std'])
                        modified_z = abs(0.6745 * (value - stats_data['median']) / stats_data['mad'])
                        
                        # Clean scores
                        z_score = np.nan_to_num(z_score, nan=0.0, posinf=1.0, neginf=0.0)
                        modified_z = np.nan_to_num(modified_z, nan=0.0, posinf=1.0, neginf=0.0)
                        
                        # Use higher deviation as contribution score
                        contribution = max(z_score / 3.0, modified_z / 3.5)
                        contribution = min(contribution, 1.0)  # Cap at 1.0
                        row_explanation[column] = float(contribution)
                
                explanations[idx] = row_explanation
        
        return explanations

class AnomalyDetectionManager:
    """Main manager for anomaly detection operations"""
    
    def __init__(self):
        self.ensemble_detector = None
        self.model_metadata = None
        self.models_dir = settings.ai.ml_storage_path / "anomaly_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def train_models(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        contamination: float = 0.1,
        hyperparameter_tuning: bool = True
    ) -> ModelMetadata:
        """Train anomaly detection models with optional hyperparameter tuning"""
        logger.info(f"Starting anomaly detection training with contamination={contamination}")
        
        # FIX: Add data validation and preprocessing
        X_processed = self._preprocess_training_data(X)
        
        if X_processed.empty:
            raise ValueError("No valid training data after preprocessing. Check your data collection pipeline.")
        
        logger.info(f"Training data processed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        
        if hyperparameter_tuning:
            contamination = self._optimize_hyperparameters(X_processed, y)
        
        # Initialize and train ensemble detector
        self.ensemble_detector = EnsembleAnomalyDetector(contamination=contamination)
        self.ensemble_detector.fit(X_processed, y)
        
        # Evaluate model performance
        performance_metrics = self._evaluate_model(X_processed, y)
        
        # Create model metadata
        self.model_metadata = ModelMetadata(
            model_name="ensemble_anomaly_detector",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            training_date=datetime.now(),
            training_samples=len(X_processed),
            features_count=X_processed.shape[1],
            performance_metrics=performance_metrics,
            contamination_rate=contamination,
            hyperparameters={"contamination": contamination},
            data_characteristics=self._analyze_data_characteristics(X_processed),
            model_path=""
        )
        
        # Save model
        model_path = self._save_model()
        self.model_metadata.model_path = model_path
        
        logger.info(f"Anomaly detection model trained and saved: {model_path}")
        return self.model_metadata
    
    def _preprocess_training_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess training data to handle issues from data collection pipeline"""
        if X.empty:
            logger.error("Empty DataFrame provided for training")
            return X
        
        logger.info(f"Original training data shape: {X.shape}")
        
        # Step 1: Handle infinite values
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        # Step 2: Get numeric columns only
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        X_numeric = X_clean[numeric_cols].copy()
        
        logger.info(f"Numeric columns: {len(numeric_cols)}")
        
        # Step 3: Handle missing values more aggressively
        # Drop columns with too many missing values (>80%)
        missing_threshold = 0.8
        columns_to_keep = []
        
        for col in X_numeric.columns:
            missing_pct = X_numeric[col].isnull().sum() / len(X_numeric)
            if missing_pct < missing_threshold:
                columns_to_keep.append(col)
            else:
                logger.warning(f"Dropping column '{col}' - {missing_pct:.1%} missing values")
        
        if not columns_to_keep:
            logger.error("No columns with sufficient data quality found")
            return pd.DataFrame()
        
        X_filtered = X_numeric[columns_to_keep].copy()
        
        # Step 4: Fill remaining missing values with different strategies
        for col in X_filtered.columns:
            if X_filtered[col].isnull().any():
                # Try forward fill first (good for time series)
                X_filtered[col] = X_filtered[col].fillna(method='ffill')
                # Then backward fill
                X_filtered[col] = X_filtered[col].fillna(method='bfill')
                # Finally use median for any remaining
                X_filtered[col] = X_filtered[col].fillna(X_filtered[col].median())
                # If still NaN (all values were NaN), fill with 0
                X_filtered[col] = X_filtered[col].fillna(0)
        
        # Step 5: Remove constant columns (no variation)
        variable_cols = []
        for col in X_filtered.columns:
            if X_filtered[col].nunique() > 1:
                variable_cols.append(col)
            else:
                logger.warning(f"Dropping constant column '{col}'")
        
        if not variable_cols:
            logger.error("No columns with variation found")
            return pd.DataFrame()
        
        X_final = X_filtered[variable_cols].copy()
        
        # Step 6: Remove outliers using IQR method to prevent extreme values
        for col in X_final.columns:
            Q1 = X_final[col].quantile(0.25)
            Q3 = X_final[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Only if there's actual spread
                lower_bound = Q1 - 3 * IQR  # More lenient than 1.5
                upper_bound = Q3 + 3 * IQR
                X_final[col] = X_final[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Step 7: Final validation
        if X_final.empty:
            logger.error("All data was filtered out during preprocessing")
            return pd.DataFrame()
        
        # Ensure we have minimum samples for training
        min_samples = 10
        if len(X_final) < min_samples:
            logger.error(f"Insufficient samples after preprocessing: {len(X_final)} < {min_samples}")
            return pd.DataFrame()
        
        logger.info(f"Preprocessed training data shape: {X_final.shape}")
        logger.info(f"Data quality: {X_final.isnull().sum().sum()} missing values, {len(X_final.columns)} features")
        
        return X_final
    
    def _preprocess_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess prediction data to match training data format"""
        if not self.ensemble_detector:
            raise ValueError("No model loaded")
        
        if data.empty:
            return data
        
        # Get expected features from training
        expected_features = self.ensemble_detector.feature_names
        current_features = list(data.columns)
        
        logger.info(f"Expected features: {len(expected_features)}, Current features: {len(current_features)}")
        
        # Add missing features with zeros
        for feature in expected_features:
            if feature not in current_features:
                data[feature] = 0.0
                logger.debug(f"Added missing feature '{feature}' with zeros")
        
        # Remove extra features not seen during training
        extra_features = [f for f in current_features if f not in expected_features]
        if extra_features:
            data = data.drop(columns=extra_features)
            logger.debug(f"Removed {len(extra_features)} extra features")
        
        # Reorder columns to match training order
        data = data.reindex(columns=expected_features, fill_value=0.0)
        
        # Clean data before prediction
        data = data.fillna(0.0)
        data = data.replace([np.inf, -np.inf], 0.0)
        
        return data
    
    async def train_models_with_fallback(
        self, 
        data_manager,
        hours_back: int = 24,
        contamination: float = 0.1,
        hyperparameter_tuning: bool = True
    ) -> ModelMetadata:
        """Train models with fallback strategies for problematic data"""
        
        # Try different data collection strategies
        strategies = [
            {"hours": hours_back, "clean": True},
            {"hours": hours_back * 2, "clean": True},  # Try more data
            {"hours": 1, "clean": False},  # Try recent data without aggressive cleaning
        ]
        
        for strategy in strategies:
            try:
                logger.info(f"Trying training strategy: {strategy}")
                
                # Collect data
                metrics_df, alerts_df = await data_manager.collect_training_data(
                    hours_back=strategy["hours"]
                )
                
                if metrics_df.empty:
                    continue
                
                # Prepare features manually instead of using the problematic pipeline
                features_df = self._prepare_simple_features(metrics_df)
                
                if not features_df.empty and len(features_df) >= 10:
                    logger.info(f"Successfully prepared {features_df.shape[0]} samples with {features_df.shape[1]} features")
                    return self.train_models(features_df, contamination=contamination, hyperparameter_tuning=hyperparameter_tuning)
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {e}")
                continue
        
        raise ValueError("All training strategies failed. Check your data collection pipeline.")

    def _prepare_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare simple features without complex rolling windows"""
        if df.empty:
            return df
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        features_df = df[numeric_cols].copy()
        
        # Simple cleaning
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Add simple time-based features if index is datetime
        if isinstance(features_df.index, pd.DatetimeIndex):
            features_df['hour'] = features_df.index.hour
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['is_weekend'] = (features_df.index.dayofweek >= 5).astype(int)
        
        return features_df
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> float:
        """Optimize hyperparameters using cross-validation"""
        logger.info("Optimizing hyperparameters...")
        
        # Test different contamination rates
        contamination_rates = [0.05, 0.1, 0.15, 0.2]
        best_score = -np.inf
        best_contamination = 0.1
        
        X_numeric = X.select_dtypes(include=[np.number])
        X_numeric = X_numeric.fillna(0).replace([np.inf, -np.inf], 0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        for contamination in contamination_rates:
            try:
                # Use Isolation Forest for hyperparameter optimization
                model = IsolationForest(contamination=contamination, random_state=42)
                
                # Cross-validation score (using outlier factor as proxy)
                scores = []
                for train_idx, val_idx in self._get_cv_splits(X_scaled):
                    model.fit(X_scaled[train_idx])
                    val_scores = model.decision_function(X_scaled[val_idx])
                    val_scores = np.nan_to_num(val_scores, nan=0.0)
                    scores.append(np.mean(val_scores))
                
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_contamination = contamination
                    
            except Exception as e:
                logger.warning(f"Error testing contamination {contamination}: {e}")
        
        logger.info(f"Best contamination rate: {best_contamination}")
        return best_contamination
    
    def _get_cv_splits(self, X: np.ndarray, n_splits: int = 3):
        """Generate cross-validation splits for time series data"""
        n_samples = len(X)
        split_size = n_samples // n_splits
        
        for i in range(n_splits):
            val_start = i * split_size
            val_end = (i + 1) * split_size if i < n_splits - 1 else n_samples
            
            train_indices = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, n_samples)
            ])
            val_indices = np.arange(val_start, val_end)
            
            yield train_indices, val_indices
    
    def _evaluate_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, float]:
        """Evaluate model performance with unsupervised metrics"""
        
        # Get predictions and scores
        predictions = self.ensemble_detector.predict(X)
        confidence_scores = self.ensemble_detector.predict_proba(X)
        
        # Clean all metrics
        metrics = {
            "anomaly_detection_rate": float(np.mean(predictions)),
            "total_anomalies_detected": int(np.sum(predictions)),
            "mean_anomaly_score": float(np.mean(confidence_scores)),
            "max_anomaly_score": float(np.max(confidence_scores)),
            "score_variance": float(np.var(confidence_scores)),
            "high_confidence_anomalies": int(np.sum(confidence_scores > 0.7)),
            "medium_confidence_anomalies": int(np.sum((confidence_scores > 0.4) & (confidence_scores <= 0.7))),
            "low_confidence_anomalies": int(np.sum(confidence_scores <= 0.4))
        }
        
        # Clean all metrics to ensure JSON compatibility
        metrics = clean_float_values(metrics)
        
        # Calculate silhouette score if we have enough samples
        try:
            from sklearn.metrics import silhouette_score
            X_numeric = X.select_dtypes(include=[np.number])
            if len(X_numeric) > 10:  # Need at least 10 samples
                X_numeric_clean = X_numeric.fillna(0).replace([np.inf, -np.inf], 0)
                X_scaled = self.ensemble_detector.scaler.transform(X_numeric_clean)
                if len(np.unique(predictions)) > 1:  # Need both normal and anomaly labels
                    silhouette = silhouette_score(X_scaled, predictions)
                    silhouette = float(np.nan_to_num(silhouette, nan=0.0))
                    metrics["silhouette_score"] = silhouette
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
        
        # Calculate stability metrics (consistency across ensemble)
        try:
            X_numeric = X.select_dtypes(include=[np.number])
            X_numeric_clean = X_numeric.fillna(0).replace([np.inf, -np.inf], 0)
            X_scaled = self.ensemble_detector.scaler.transform(X_numeric_clean)
            
            # Get individual model predictions
            if_pred = self.ensemble_detector.models['isolation_forest'].predict(X_scaled)
            lof_pred = self.ensemble_detector.models['lof'].predict(X_scaled)
            
            # Convert to binary (1 for anomaly, 0 for normal)
            if_binary = (if_pred == -1).astype(int)
            lof_binary = (lof_pred == -1).astype(int)
            
            # Calculate agreement between models
            agreement = float(np.mean(if_binary == lof_binary))
            metrics["ensemble_agreement"] = agreement
            
        except Exception as e:
            logger.warning(f"Could not calculate ensemble agreement: {e}")
        
        # If supervised labels are available, calculate traditional metrics
        if y is not None:
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                
                supervised_metrics = {
                    "supervised_precision": float(precision_score(y, predictions, zero_division=0)),
                    "supervised_recall": float(recall_score(y, predictions, zero_division=0)),
                    "supervised_f1_score": float(f1_score(y, predictions, zero_division=0)),
                    "supervised_accuracy": float(accuracy_score(y, predictions))
                }
                supervised_metrics = clean_float_values(supervised_metrics)
                metrics.update(supervised_metrics)
                
            except Exception as e:
                logger.warning(f"Could not calculate supervised metrics: {e}")
        
        return metrics
    
    def _analyze_data_characteristics(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of training data"""
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        characteristics = {
            "feature_count": len(numeric_columns),
            "sample_count": len(X),
            "missing_values_pct": (X.isnull().sum().sum() / (len(X) * len(X.columns))) * 100,
            "numeric_features": list(numeric_columns),
            "feature_statistics": {}
        }
        
        # Calculate feature statistics with proper cleaning
        for col in numeric_columns:
            col_data = X[col].dropna()
            if len(col_data) > 0:
                characteristics["feature_statistics"][col] = {
                    "mean": float(np.nan_to_num(col_data.mean(), nan=0.0)),
                    "std": float(np.nan_to_num(col_data.std(), nan=0.0)),
                    "min": float(np.nan_to_num(col_data.min(), nan=0.0)),
                    "max": float(np.nan_to_num(col_data.max(), nan=0.0))
                }
        
        return clean_float_values(characteristics)
    
    def _save_model(self) -> str:
        """Save trained model and metadata"""
        if not self.ensemble_detector or not self.model_metadata:
            raise ValueError("No model to save")
        
        # Create versioned filename
        timestamp = self.model_metadata.version
        model_filename = f"anomaly_detector_{timestamp}.joblib"
        metadata_filename = f"anomaly_detector_{timestamp}_metadata.json"
        
        model_path = self.models_dir / model_filename
        metadata_path = self.models_dir / metadata_filename
        
        # Backup existing model if it exists
        self._backup_existing_models()
        
        # Save model
        joblib.dump({
            'ensemble_detector': self.ensemble_detector,
            'model_metadata': self.model_metadata
        }, model_path)
        
        # Save metadata separately for easy access
        metadata_dict = asdict(self.model_metadata)
        metadata_dict['training_date'] = self.model_metadata.training_date.isoformat()
        metadata_dict = clean_float_values(metadata_dict)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        return str(model_path)
    
    def _backup_existing_models(self):
        """Backup existing models before saving new ones"""
        backup_dir = self.models_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Move existing models to backup
        for model_file in self.models_dir.glob("anomaly_detector_*.joblib"):
            if model_file.parent != backup_dir:
                backup_path = backup_dir / model_file.name
                model_file.rename(backup_path)
                logger.info(f"Backed up model: {backup_path}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load a trained model"""
        if model_path is None:
            # Load latest model
            model_files = list(self.models_dir.glob("anomaly_detector_*.joblib"))
            if not model_files:
                logger.error("No trained models found")
                return False
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        try:
            saved_data = joblib.load(model_path)
            self.ensemble_detector = saved_data['ensemble_detector']
            self.model_metadata = saved_data['model_metadata']
            
            logger.info(f"Model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return False
    
    def detect_anomalies_realtime(
        self, 
        data: pd.DataFrame,
        return_explanations: bool = True
    ) -> Dict[str, Any]:
        """Detect anomalies in real-time data"""
        if not self.ensemble_detector:
            raise ValueError("No model loaded. Train or load a model first.")
        
        start_time = datetime.now()
        
        # FIX: Add data preprocessing for prediction data
        data_processed = self._preprocess_prediction_data(data)
        
        if data_processed.empty:
            logger.warning("No valid data for anomaly detection after preprocessing")
            return {
                "timestamp": datetime.now().isoformat(),
                "total_samples": 0,
                "anomaly_count": 0,
                "anomaly_rate": 0.0,
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "anomalies": [],
                "error": "No valid data after preprocessing"
            }
        
        # Get predictions and confidence scores
        predictions = self.ensemble_detector.predict(data_processed)
        confidence_scores = self.ensemble_detector.predict_proba(data_processed)
        
        # Find anomalous indices
        anomaly_indices = np.where(predictions == 1)[0].tolist()
        
        # Get explanations if requested
        explanations = {}
        if return_explanations and anomaly_indices:
            explanations = self.ensemble_detector.explain_anomalies(data_processed, anomaly_indices)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create detailed results
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(data_processed),
            "anomaly_count": len(anomaly_indices),
            "anomaly_rate": len(anomaly_indices) / len(data_processed) if len(data_processed) > 0 else 0.0,
            "processing_time_seconds": processing_time,
            "anomalies": []
        }
        
        # Add detailed anomaly information
        for idx in anomaly_indices:
            confidence_score = float(confidence_scores[idx])
            confidence_score = np.nan_to_num(confidence_score, nan=0.0, posinf=1.0, neginf=0.0)
            
            anomaly_info = {
                "index": int(idx),
                "confidence_score": confidence_score,
                "severity": self._calculate_severity(confidence_score),
                "timestamp": data_processed.index[idx].isoformat() if hasattr(data_processed.index[idx], 'isoformat') else str(data_processed.index[idx]),
                "affected_metrics": clean_float_values(explanations.get(idx, {}))
            }
            results["anomalies"].append(anomaly_info)
        
        # Sort by confidence score (highest first)
        results["anomalies"].sort(key=lambda x: x["confidence_score"], reverse=True)
        
        # Final cleaning of results
        results = clean_float_values(results)
        
        logger.info(f"Real-time anomaly detection completed: {len(anomaly_indices)} anomalies found in {processing_time:.3f}s")
        return results
    
    def _calculate_severity(self, confidence_score: float) -> str:
        """Calculate severity level based on confidence score"""
        confidence_score = np.nan_to_num(confidence_score, nan=0.0)
        
        if confidence_score >= 0.8:
            return "CRITICAL"
        elif confidence_score >= 0.6:
            return "HIGH"
        elif confidence_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and metadata"""
        if not self.ensemble_detector or not self.model_metadata:
            return {"status": "no_model_loaded"}
        
        status = {
            "status": "model_loaded",
            "model_name": self.model_metadata.model_name,
            "version": self.model_metadata.version,
            "training_date": self.model_metadata.training_date.isoformat(),
            "training_samples": self.model_metadata.training_samples,
            "features_count": self.model_metadata.features_count,
            "performance_metrics": self.model_metadata.performance_metrics,
            "contamination_rate": self.model_metadata.contamination_rate
        }
        
        return clean_float_values(status)

# Global anomaly detection manager instance
anomaly_manager = AnomalyDetectionManager()

async def test_anomaly_detection():
    """Test function for anomaly detection functionality"""
    try:
        from data_collector import DataCollectionManager
        
        # Collect training data
        data_manager = DataCollectionManager()
        metadata = await anomaly_manager.train_models_with_fallback(
            data_manager, 
            hours_back=24,
            contamination=0.1,
            hyperparameter_tuning=False
        )
        
        # Print results with proper unsupervised metrics
        print("\n" + "="*50)
        print("ANOMALY DETECTION TEST RESULTS")
        print("="*50)
        print(f"Model Version: {metadata.version}")
        print(f"Training Samples: {metadata.training_samples}")
        print(f"Features: {metadata.features_count}")
        
        # Print unsupervised metrics
        print("\nUnsupervised Performance Metrics:")
        perf = metadata.performance_metrics
        print(f"  - Anomaly Detection Rate: {perf.get('anomaly_detection_rate', 0):.2%}")
        print(f"  - Mean Anomaly Score: {perf.get('mean_anomaly_score', 0):.3f}")
        print(f"  - Ensemble Agreement: {perf.get('ensemble_agreement', 0):.2%}")
        print(f"  - High Confidence Anomalies: {perf.get('high_confidence_anomalies', 0)}")
        
        if 'silhouette_score' in perf:
            print(f"  - Silhouette Score: {perf['silhouette_score']:.3f}")
        
        # Print supervised metrics if available
        if 'supervised_f1_score' in perf:
            print(f"\nSupervised Metrics (if labels available):")
            print(f"  - Precision: {perf.get('supervised_precision', 0):.3f}")
            print(f"  - Recall: {perf.get('supervised_recall', 0):.3f}")
            print(f"  - F1-Score: {perf.get('supervised_f1_score', 0):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Anomaly detection test failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_anomaly_detection())
