import asyncio
import aiomysql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import logging
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollectionError(Exception):
    """Custom exception for data collection errors"""
    pass

class InfluxDBCollector:
    """Async InfluxDB data collection for historical metrics"""
    
    def __init__(self):
        self.client = None
        self.connection_params = settings.get_influxdb_connection_params()
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = InfluxDBClientAsync(**self.connection_params)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.close()
# Fix the query_historical_metrics method around line 40

    async def query_historical_metrics(
        self, 
        time_range_hours: int = 24,
        host_filter: str = None,
        measurements: List[str] = None
    ) -> pd.DataFrame:
        """
        Query historical metrics from InfluxDB with time range and filtering
        
        Args:
            time_range_hours: Hours of historical data to retrieve
            host_filter: Specific host to filter (defaults to target_host)
            measurements: Specific measurements to retrieve
            
        Returns:
            DataFrame with historical metrics data
        """
        if not host_filter:
            host_filter = settings.monitoring.target_host
            
        if not measurements:
            measurements = settings.monitoring.key_metrics
            
        try:
            # Build Flux query for multiple measurements
            measurement_filters = " or ".join([
                f'r["_measurement"] == "{m}"' for m in measurements
            ])
            
            flux_query = f'''
            from(bucket: "{self.connection_params["bucket"]}")
                |> range(start: -{time_range_hours}h)
                |> filter(fn: (r) => {measurement_filters})
                |> filter(fn: (r) => r["_field"] == "value")
                |> filter(fn: (r) => r["host"] == "{host_filter}")
                |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''
            
            query_api = self.client.query_api()
            result = await query_api.query_data_frame(flux_query)
            
            # Fix: Handle both list and DataFrame results
            if isinstance(result, list):
                if len(result) == 0:
                    logger.warning(f"No data found for host {host_filter} in last {time_range_hours} hours")
                    return pd.DataFrame()
                # If list contains DataFrames, concatenate them
                if all(isinstance(df, pd.DataFrame) for df in result):
                    result = pd.concat(result, ignore_index=True)
                else:
                    logger.warning("InfluxDB returned unexpected list format")
                    return pd.DataFrame()
            
            if result.empty:
                logger.warning(f"No data found for host {host_filter} in last {time_range_hours} hours")
                return pd.DataFrame()
            
            # Clean and process the data
            df = self._process_influx_dataframe(result)
            logger.info(f"Retrieved {len(df)} records for {host_filter}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error querying InfluxDB: {e}")
            raise DataCollectionError(f"InfluxDB query failed: {e}")


    
# Fix the query_metric_aggregations method 

    async def query_metric_aggregations(
        self,
        measurement: str,
        aggregation: str = "mean",
        window: str = "5m",
        time_range_hours: int = 24
    ) -> pd.DataFrame:
        """
        Query aggregated metrics with configurable time windows
        
        Args:
            measurement: Specific measurement to aggregate
            aggregation: Aggregation function (mean, max, min, last)
            window: Time window for aggregation (e.g., "5m", "1h")
            time_range_hours: Hours of data to retrieve
            
        Returns:
            DataFrame with aggregated data
        """
        try:
            flux_query = f'''
            from(bucket: "{self.connection_params["bucket"]}")
                |> range(start: -{time_range_hours}h)
                |> filter(fn: (r) => r["_measurement"] == "{measurement}")
                |> filter(fn: (r) => r["_field"] == "value")
                |> filter(fn: (r) => r["host"] == "{settings.monitoring.target_host}")
                |> aggregateWindow(every: {window}, fn: {aggregation}, createEmpty: false)
                |> yield(name: "{aggregation}")
            '''
            
            query_api = self.client.query_api()
            result = await query_api.query_data_frame(flux_query)
            
            # Fix: Handle both list and DataFrame results
            if isinstance(result, list):
                if len(result) == 0:
                    return pd.DataFrame()
                # If list contains DataFrames, concatenate them
                if all(isinstance(df, pd.DataFrame) for df in result):
                    result = pd.concat(result, ignore_index=True)
                else:
                    return pd.DataFrame()
            
            if not result.empty:
                df = self._process_influx_dataframe(result)
                df['aggregation_type'] = aggregation
                df['window_size'] = window
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error querying aggregated data: {e}")
            raise DataCollectionError(f"Aggregation query failed: {e}")
    
    def _process_influx_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean InfluxDB DataFrame"""
        if df.empty:
            return df
            
        # Rename time column and set as index
        if '_time' in df.columns:
            df = df.rename(columns={'_time': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Remove InfluxDB metadata columns
        metadata_cols = ['result', 'table', '_start', '_stop']
        df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
        
        # Convert numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_index()
        
        return df

class MySQLCollector:
    """Async MySQL data collection for alert and event data"""
    
    def __init__(self):
        self.pool = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        try:
            self.pool = await aiomysql.create_pool(
                host=settings.database.mysql_host,
                port=settings.database.mysql_port,
                user=settings.database.mysql_user,
                password=settings.database.mysql_password,
                db=settings.database.mysql_database,
                autocommit=True,
                minsize=1,
                maxsize=10
            )
            return self
        except Exception as e:
            logger.error(f"Failed to create MySQL connection pool: {e}")
            raise DataCollectionError(f"MySQL connection failed: {e}")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
    
    async def query_historical_alerts(
        self, 
        days_back: int = 7,
        host_filter: str = None
    ) -> pd.DataFrame:
        """
        Query historical alerts with trigger and host information
        
        Args:
            days_back: Number of days of alert history to retrieve
            host_filter: Specific host to filter alerts
            
        Returns:
            DataFrame with alert history and metadata
        """
        if not host_filter:
            host_filter = settings.monitoring.target_host
            
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        query = """
        SELECT 
            a.alertid,
            a.clock as alert_time,
            a.sendto,
            a.subject,
            a.message,
            a.status as alert_status,
            e.eventid,
            e.clock as event_time,
            e.value as event_value,
            e.severity,
            e.acknowledged,
            t.triggerid,
            t.description as trigger_description,
            t.priority as trigger_priority,
            t.status as trigger_status,
            h.hostid,
            h.host as hostname,
            h.name as host_display_name,
            h.status as host_status
        FROM alerts a
        JOIN events e ON a.eventid = e.eventid
        JOIN triggers t ON e.objectid = t.triggerid
        JOIN functions f ON t.triggerid = f.triggerid
        JOIN items i ON f.itemid = i.itemid
        JOIN hosts h ON i.hostid = h.hostid
        WHERE a.clock >= %s
          AND h.host = %s
          AND e.source = 0
        ORDER BY a.clock DESC
        """
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(query, (start_time, host_filter))
                    results = await cursor.fetchall()
                    
            if not results:
                logger.warning(f"No alerts found for host {host_filter}")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            df = self._process_mysql_dataframe(df)
            
            logger.info(f"Retrieved {len(df)} alerts for {host_filter}")
            return df
            
        except Exception as e:
            logger.error(f"Error querying MySQL alerts: {e}")
            raise DataCollectionError(f"MySQL alert query failed: {e}")
    
    async def query_trigger_events(
        self, 
        days_back: int = 7,
        severity_filter: int = None
    ) -> pd.DataFrame:
        """
        Query trigger events with severity filtering
        
        Args:
            days_back: Number of days of events to retrieve
            severity_filter: Minimum severity level (0-5)
            
        Returns:
            DataFrame with trigger events
        """
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        severity_condition = ""
        params = [start_time, settings.monitoring.target_host]
        
        if severity_filter is not None:
            severity_condition = "AND e.severity >= %s"
            params.append(severity_filter)
        
        query = f"""
        SELECT 
            e.eventid,
            e.clock as event_time,
            e.value as event_value,
            e.severity,
            e.acknowledged,
            t.triggerid,
            t.description as trigger_description,
            t.expression as trigger_expression,
            t.priority,
            h.host as hostname,
            COUNT(*) OVER (PARTITION BY t.triggerid) as trigger_frequency
        FROM events e
        JOIN triggers t ON e.objectid = t.triggerid
        JOIN functions f ON t.triggerid = f.triggerid
        JOIN items i ON f.itemid = i.itemid
        JOIN hosts h ON i.hostid = h.hostid
        WHERE e.clock >= %s
          AND h.host = %s
          AND e.source = 0
          {severity_condition}
        ORDER BY e.clock DESC
        """
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(query, params)
                    results = await cursor.fetchall()
            
            if not results:
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            df = self._process_mysql_dataframe(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error querying trigger events: {e}")
            raise DataCollectionError(f"MySQL events query failed: {e}")
    
    def _process_mysql_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean MySQL DataFrame"""
        if df.empty:
            return df
        
        # Convert timestamp columns
        timestamp_columns = ['alert_time', 'event_time']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit='s')
        
        # Convert numeric columns
        numeric_columns = ['severity', 'priority', 'trigger_frequency']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

class DataPreprocessor:
    """Data preprocessing and feature engineering pipeline"""
    
    def __init__(self):
        self.scalers = {}
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling outliers, missing values, and inconsistencies
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                # Use forward fill then backward fill for time series data
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still missing, use median
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Detect and handle outliers using IQR method
        for col in numeric_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
# Optimize the create_features method to reduce fragmentation

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for ML models (optimized version)
        """
        if df.empty:
            return df
            
        df_features = df.copy()
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        
        # Pre-allocate dictionary for new features to avoid fragmentation
        new_features = {}
        
        # Rolling statistics for each numeric column
        for col in numeric_columns:
            for window in settings.ai.rolling_window_sizes:
                new_features[f'{col}_rolling_mean_{window}m'] = (
                    df_features[col].rolling(window=f'{window}min').mean()
                )
                new_features[f'{col}_rolling_std_{window}m'] = (
                    df_features[col].rolling(window=f'{window}min').std()
                )
                new_features[f'{col}_rolling_max_{window}m'] = (
                    df_features[col].rolling(window=f'{window}min').max()
                )
                new_features[f'{col}_rolling_min_{window}m'] = (
                    df_features[col].rolling(window=f'{window}min').min()
                )
        
        # Time-based features
        if df_features.index.name == 'timestamp' or 'timestamp' in df_features.columns:
            time_col = df_features.index if df_features.index.name == 'timestamp' else df_features['timestamp']
            
            new_features['hour_of_day'] = time_col.hour
            new_features['day_of_week'] = time_col.dayofweek
            new_features['is_weekend'] = (time_col.dayofweek >= 5).astype(int)
            new_features['is_business_hours'] = (
                (time_col.hour >= 9) & (time_col.hour <= 17) & (time_col.dayofweek < 5)
            ).astype(int)
        
        # Rate of change and lag features
        for col in numeric_columns:
            new_features[f'{col}_rate_change'] = df_features[col].pct_change()
            new_features[f'{col}_diff'] = df_features[col].diff()
            
            for lag in settings.ai.feature_lag_periods:
                new_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
        
        # Cross-correlations between key metrics
        if len(numeric_columns) > 1:
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    new_features[f'{col1}_{col2}_ratio'] = (
                        df_features[col1] / (df_features[col2] + 1e-8)
                    )
        
        # Add all new features at once to avoid fragmentation
        feature_df = pd.DataFrame(new_features, index=df_features.index)
        df_features = pd.concat([df_features, feature_df], axis=1)
        
        # Remove infinite and NaN values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.dropna()
        
        # Fix deprecated fillna usage
        df_features = df_features.ffill().bfill()
        
        logger.info(f"Feature engineering completed. Features: {df_features.shape[1]}")
        return df_features
    
    

    
    def normalize_data(
        self, 
        df: pd.DataFrame, 
        method: str = "standard",
        fit_transform: bool = True
    ) -> pd.DataFrame:
        """
        Normalize/scale numerical features
        
        Args:
            df: Input DataFrame
            method: Scaling method ("standard" or "minmax")
            fit_transform: Whether to fit the scaler or use existing one
            
        Returns:
            Normalized DataFrame
        """
        if df.empty:
            return df
        
        df_norm = df.copy()
        numeric_columns = df_norm.select_dtypes(include=[np.number]).columns
        
        if method not in self.scalers:
            if method == "standard":
                self.scalers[method] = StandardScaler()
            elif method == "minmax":
                self.scalers[method] = MinMaxScaler()
            else:
                raise ValueError("Method must be 'standard' or 'minmax'")
        
        scaler = self.scalers[method]
        
        if fit_transform:
            scaled_data = scaler.fit_transform(df_norm[numeric_columns])
        else:
            scaled_data = scaler.transform(df_norm[numeric_columns])
        
        df_norm[numeric_columns] = scaled_data
        
        logger.info(f"Data normalization completed using {method} scaling")
        return df_norm
    
    def calculate_data_quality_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate data quality metrics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {"completeness": 0.0, "consistency": 0.0, "accuracy": 0.0}
        
        # Completeness: percentage of non-null values
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        
        # Consistency: check for data type consistency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        consistency_score = len(numeric_cols) / len(df.columns) * 100
        
        # Accuracy: detect outliers using Isolation Forest
        if len(numeric_cols) > 0:
            outlier_data = df[numeric_cols].fillna(0)
            if len(outlier_data) > 10:  # Need minimum samples for outlier detection
                outliers = self.outlier_detector.fit_predict(outlier_data)
                accuracy = (np.sum(outliers == 1) / len(outliers)) * 100
            else:
                accuracy = 100.0
        else:
            accuracy = 100.0
        
        quality_metrics = {
            "completeness": round(completeness, 2),
            "consistency": round(consistency_score, 2), 
            "accuracy": round(accuracy, 2),
            "overall_score": round((completeness + consistency_score + accuracy) / 3, 2)
        }
        
        logger.info(f"Data quality assessment: {quality_metrics}")
        return quality_metrics

class DataCollectionManager:
    """Main data collection coordinator"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        
    async def collect_training_data(
        self, 
        hours_back: int = 168,  # 1 week
        include_alerts: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect comprehensive training data from both InfluxDB and MySQL
        
        Args:
            hours_back: Hours of historical data to collect
            include_alerts: Whether to include alert/event data
            
        Returns:
            Tuple of (metrics_df, alerts_df)
        """
        logger.info(f"Starting data collection for last {hours_back} hours")
        
        # Collect metrics data from InfluxDB
        async with InfluxDBCollector() as influx_collector:
            metrics_df = await influx_collector.query_historical_metrics(
                time_range_hours=hours_back
            )
        
        alerts_df = pd.DataFrame()
        if include_alerts:
            # Collect alerts data from MySQL
            async with MySQLCollector() as mysql_collector:
                alerts_df = await mysql_collector.query_historical_alerts(
                    days_back=int(hours_back / 24) + 1
                )
        
        # Data quality assessment
        if not metrics_df.empty:
            quality_score = self.preprocessor.calculate_data_quality_score(metrics_df)
            logger.info(f"Metrics data quality score: {quality_score['overall_score']}%")
        
        logger.info(f"Data collection completed: {len(metrics_df)} metrics, {len(alerts_df)} alerts")
        return metrics_df, alerts_df
    
    async def prepare_ml_dataset(
        self, 
        metrics_df: pd.DataFrame,
        alerts_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Prepare final ML-ready dataset with features and labels
        
        Args:
            metrics_df: Metrics DataFrame from InfluxDB
            alerts_df: Alerts DataFrame from MySQL
            
        Returns:
            ML-ready DataFrame with features and labels
        """
        if metrics_df.empty:
            logger.warning("No metrics data available for ML dataset preparation")
            return pd.DataFrame()
        
        # Clean and preprocess metrics data
        clean_metrics = self.preprocessor.clean_data(metrics_df)
        feature_metrics = self.preprocessor.create_features(clean_metrics)
        normalized_metrics = self.preprocessor.normalize_data(feature_metrics)
        
        # Add anomaly labels if alerts data is available
        if alerts_df is not None and not alerts_df.empty:
            normalized_metrics = self._add_anomaly_labels(normalized_metrics, alerts_df)
        else:
            # Create binary labels based on statistical anomalies
            normalized_metrics['is_anomaly'] = 0  # Default to normal
        
        logger.info(f"ML dataset prepared with shape: {normalized_metrics.shape}")
        return normalized_metrics
    
    def _add_anomaly_labels(
        self, 
        metrics_df: pd.DataFrame, 
        alerts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add anomaly labels to metrics based on alert timestamps
        
        Args:
            metrics_df: Metrics DataFrame
            alerts_df: Alerts DataFrame
            
        Returns:
            Metrics DataFrame with anomaly labels
        """
        df_labeled = metrics_df.copy()
        df_labeled['is_anomaly'] = 0
        
        if alerts_df.empty:
            return df_labeled
        
        # Convert alert times to timestamp index
        alert_times = pd.to_datetime(alerts_df['alert_time'])
        
        # Mark periods around alerts as anomalous (±30 minutes)
        for alert_time in alert_times:
            start_time = alert_time - timedelta(minutes=30)
            end_time = alert_time + timedelta(minutes=30)
            
            mask = (df_labeled.index >= start_time) & (df_labeled.index <= end_time)
            df_labeled.loc[mask, 'is_anomaly'] = 1
        
        anomaly_count = df_labeled['is_anomaly'].sum()
        logger.info(f"Added anomaly labels: {anomaly_count} anomalous periods identified")
        
        return df_labeled

# Usage example and testing
async def test_data_collection():
    """Test function to validate data collection functionality"""
    try:
        manager = DataCollectionManager()
        
        # Test data collection
        metrics_df, alerts_df = await manager.collect_training_data(hours_back=24)
        
        print(f"Collected {len(metrics_df)} metric records")
        print(f"Collected {len(alerts_df)} alert records")
        
        if not metrics_df.empty:
            print(f"Metrics columns: {list(metrics_df.columns)}")
            print(f"Metrics time range: {metrics_df.index.min()} to {metrics_df.index.max()}")
        
        # Test ML dataset preparation
        ml_dataset = await manager.prepare_ml_dataset(metrics_df, alerts_df)
        
        if not ml_dataset.empty:
            print(f"ML dataset shape: {ml_dataset.shape}")
            print(f"Features: {ml_dataset.shape[1]}")
            if 'is_anomaly' in ml_dataset.columns:
                print(f"Anomaly ratio: {ml_dataset['is_anomaly'].mean():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data collection test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test
    asyncio.run(test_data_collection())





