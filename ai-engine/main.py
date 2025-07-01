
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from config import settings, validate_configuration
from data_collector import DataCollectionManager, test_data_collection
import logging
import numpy as np
import random
from typing import List, Dict, Any

from anomaly_detection import anomaly_manager
from typing import Optional

from prediction_engine import prediction_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered monitoring and anomaly detection for Zabbix metrics"
)

# Add CORS middleware for Grafana
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data collection manager
data_manager = DataCollectionManager()

# Global variables for AI simulation
current_anomalies = 2
system_risk_level = 25
model_accuracy = 85.5
model_status = 1

async def auto_generate_predictions():
    """Automatically generate predictions for all trained models"""
    while True:
        try:
            logger.info("Auto-generating predictions for all trained models...")
            
            # Get all trained models
            status = prediction_manager.get_prediction_status()
            trained_metrics = status.get("metrics", [])
            
            if trained_metrics:
                logger.info(f"Generating forecasts for: {trained_metrics}")
                
                # Generate forecasts for all trained models
                forecasts = prediction_manager.generate_forecasts(
                    forecast_horizon=24,
                    metrics=trained_metrics
                )
                
                successful_forecasts = len([
                    metric for metric, data in forecasts.get("forecasts", {}).items()
                    if "error" not in data
                ])
                
                logger.info(f"Generated forecasts for {successful_forecasts}/{len(trained_metrics)} metrics")
            else:
                logger.warning("No trained prediction models found")
            
        except Exception as e:
            logger.error(f"Auto-prediction error: {e}")
        
        # Wait 30 minutes before next auto-generation
        await asyncio.sleep(1800)

# Add this function after your existing auto_generate_predictions function

async def auto_train_prediction_models():
    """Automatically retrain prediction models periodically"""
    while True:
        try:
            logger.info("Auto-training prediction models...")
            
            # Define the metrics you want to train (same as your curl command)
            target_metrics = [
                "system_cpu_util",
                "system_cpu_util_user", 
                "system_cpu_load_all_avg1",
                "vm_memory_utilization"
            ]
            
            # Check if retraining is needed
            status = prediction_manager.get_prediction_status()
            current_models = status.get("models", {})
            
            should_retrain = False
            
            # Check if any models are missing or old
            for metric in target_metrics:
                if metric not in current_models:
                    logger.info(f"Missing model for {metric}, will retrain")
                    should_retrain = True
                    break
                else:
                    # Check model age (retrain if older than 4 hours)
                    model_info = current_models[metric]
                    version = model_info.get("version", "")
                    if version and "_" in version:
                        try:
                            # Extract timestamp from version (format: 1.0_YYYYMMDD_HHMMSS)
                            parts = version.split("_")
                            if len(parts) >= 3:
                                timestamp_str = f"{parts[-2]}_{parts[-1]}"
                                model_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                                hours_old = (datetime.now() - model_time).total_seconds() / 3600
                                
                                if hours_old > 4:  # Retrain if older than 4 hours
                                    logger.info(f"Model {metric} is {hours_old:.1f} hours old, will retrain")
                                    should_retrain = True
                                    break
                        except Exception as e:
                            logger.warning(f"Could not parse model timestamp for {metric}: {e}")
                            should_retrain = True  # Retrain if we can't determine age
                            break
            
            if should_retrain:
                logger.info("Starting automatic model training...")
                
                # Collect training data (same parameters as your curl command)
                metrics_df, _ = await data_manager.collect_training_data(
                    hours_back=168,  # 1 week
                    include_alerts=False
                )
                
                if not metrics_df.empty:
                    # Use clean metrics
                    clean_metrics = data_manager.preprocessor.clean_data(metrics_df)
                    
                    # Train prediction models (same as your curl command)
                    results = prediction_manager.train_prediction_models(
                        clean_metrics,
                        target_metrics=target_metrics,
                        forecast_horizon=24
                    )
                    
                    if results:
                        logger.info(f"[OK] Auto-training completed: {len(results)} models trained")
                        
                        # Log training results
                        for metric, metadata in results.items():
                            mae_prophet = metadata.performance_metrics.get("prophet_mae", 0)
                            mae_arima = metadata.performance_metrics.get("arima_mae", 0)
                            logger.info(f"  - {metric}: {metadata.training_samples} samples, Prophet MAE: {mae_prophet:.2f}, ARIMA MAE: {mae_arima:.2f}")
                    else:
                        logger.warning("[ERROR] Auto-training failed: No models were trained")
                else:
                    logger.warning("[WARNING] Auto-training skipped: No training data available")
            else:
                logger.info("[OK] Auto-training skipped: All models are current")
            
        except Exception as e:
            logger.error(f"[ERROR] Auto-training error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Wait 30 minutes before next check 
        logger.info(" Next auto-training check in 30 minutes...")
        await asyncio.sleep(1800)


@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Validate configuration
    if not validate_configuration():
        logger.error("Configuration validation failed")
        raise RuntimeError("Invalid configuration")
    
    # Start auto-prediction task
    asyncio.create_task(auto_generate_predictions())
    
    # Start auto-training task
    asyncio.create_task(auto_train_prediction_models())
    
    logger.info("AI Engine started successfully with auto-training enabled")

# ==== GRAFANA SIMPLEJSON DATASOURCE ENDPOINTS ====

@app.get("/")
async def datasource_test():
    """Test datasource connection for Grafana"""
    return {"status": "success", "message": "AI Engine is running"}

@app.post("/search")
async def search_metrics(request: Request):
    try:
        # Get the actual trained models from prediction status
        status_response = prediction_manager.get_prediction_status()
        
        # Extract model names from the status response structure you showed
        trained_models = []
        if "models" in status_response:
            trained_models = list(status_response["models"].keys())
        
        # Create prediction targets based on your actual trained models
        available_targets = []
        
        for model_name in trained_models:
            # Map your actual model names to Grafana targets
            if "cpu" in model_name.lower():
                if "system_cpu_util" in model_name:
                    available_targets.append("prediction_system_cpu_util")
                elif "system_cpu_load" in model_name:
                    available_targets.append("prediction_cpu_load_avg")
            elif "memory" in model_name.lower():
                available_targets.append("prediction_memory_util")
        
        # Remove duplicates and add static targets
        available_targets = list(set(available_targets))
        available_targets.extend([
            "anomalies_current",
            "system_risk_level", 
            "model_health_score",
            "model_status",
            "model_accuracy"
        ])
        
        logger.info(f"Available prediction targets: {available_targets}")
        logger.info(f"Based on trained models: {trained_models}")
        
        return available_targets
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return []

@app.post("/query")
async def query_metrics(request: Request):
    try:
        data = await request.json()
        targets = data.get("targets", [])
        time_range = data.get("range", {})
        
        logger.info(f"Query request received: targets={[t.get('target') for t in targets]}")
        
        results = []
        
        for target in targets:
            target_name = target.get("target", "")
            logger.info(f"Processing target: {target_name}")
            
            if target_name.startswith("prediction_"):
                # Map Grafana targets back to actual model names
                actual_model_name = None
                
                # Get the current trained models to ensure we're using the right names
                status_response = prediction_manager.get_prediction_status()
                trained_models = []
                if "models" in status_response:
                    trained_models = list(status_response["models"].keys())
                
                logger.info(f"Available trained models: {trained_models}")
                
                # Map targets to actual model names
                if target_name == "prediction_system_cpu_util":
                    for model in trained_models:
                        if "system_cpu_util" in model:
                            actual_model_name = model
                            break
                            
                elif target_name == "prediction_memory_util":
                    for model in trained_models:
                        if "memory" in model.lower():
                            actual_model_name = model
                            break
                            
                elif target_name == "prediction_cpu_load_avg":
                    for model in trained_models:
                        if "system_cpu_load" in model:
                            actual_model_name = model
                            break
                
                logger.info(f"Mapped {target_name} to {actual_model_name}")
                
                if actual_model_name:
                    try:
                        logger.info(f"Generating forecast for {actual_model_name}")
                        forecasts = prediction_manager.generate_forecasts(
                            forecast_horizon=24,
                            metrics=[actual_model_name]
                        )
                        
                        logger.info(f"Forecast response keys: {forecasts.keys() if isinstance(forecasts, dict) else 'Not a dict'}")
                        
                        # FIX: Check for forecasts data directly, not status field
                        if isinstance(forecasts, dict) and "forecasts" in forecasts:
                            forecast_data = forecasts.get("forecasts", {})
                            
                            if actual_model_name in forecast_data:
                                metric_forecast = forecast_data[actual_model_name]
                                models = metric_forecast.get("models", {})
                                
                                # Try Prophet first, then ARIMA
                                model_data = None
                                model_type = None
                                
                                if "prophet" in models and models["prophet"]:
                                    model_data = models["prophet"]
                                    model_type = "Prophet"
                                elif "arima" in models and models["arima"]:
                                    model_data = models["arima"]
                                    model_type = "ARIMA"
                                
                                logger.info(f"Using {model_type} model for {actual_model_name}")
                                
                                if model_data:
                                    timestamps = model_data.get("timestamps", [])
                                    values = model_data.get("predicted_values", [])
                                    
                                    logger.info(f"Model data: {len(timestamps)} timestamps, {len(values)} values")
                                    
                                    if timestamps and values and len(timestamps) == len(values):
                                        datapoints = []
                                        
                                        for timestamp, value in zip(timestamps, values):
                                            try:
                                                # Convert timestamp to milliseconds
                                                if isinstance(timestamp, str):
                                                    ts = int(pd.to_datetime(timestamp).timestamp() * 1000)
                                                else:
                                                    ts = int(timestamp * 1000)
                                                
                                                # Ensure value is a valid number
                                                val = float(value) if value is not None else 0.0
                                                datapoints.append([val, ts])
                                                
                                            except Exception as e:
                                                logger.warning(f"Failed to convert datapoint {timestamp}, {value}: {e}")
                                                continue
                                        
                                        if datapoints:
                                            # Create a readable target name
                                            display_name = target_name.replace('prediction_', '').replace('_', ' ').title()
                                            display_name += f" Forecast ({model_type})"
                                            
                                            results.append({
                                                "target": display_name,
                                                "datapoints": datapoints
                                            })
                                            
                                            logger.info(f"Generated {len(datapoints)} forecast points for {actual_model_name}")
                                        else:
                                            logger.warning(f"No valid datapoints generated for {actual_model_name}")
                                    else:
                                        logger.warning(f"Invalid timestamps/values for {actual_model_name}: {len(timestamps)} timestamps, {len(values)} values")
                                else:
                                    logger.warning(f"No model data available for {actual_model_name}")
                            else:
                                logger.warning(f"Metric {actual_model_name} not in forecast data. Available: {list(forecast_data.keys())}")
                        else:
                            logger.error(f"Invalid forecast response format: {type(forecasts)}")
                            
                    except Exception as e:
                        logger.error(f"Error processing prediction for {target_name}: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.warning(f"No matching trained model found for {target_name}")
            
            # Handle static targets (keep your existing logic)
            elif target_name == "anomalies_current":
                global current_anomalies
                current_anomalies = random.randint(0, 8)
                results.append({
                    "target": "Current Anomalies",
                    "datapoints": [[current_anomalies, int(datetime.now().timestamp() * 1000)]]
                })
            
            elif target_name == "system_risk_level":
                global system_risk_level
                system_risk_level = random.randint(10, 75)
                results.append({
                    "target": "System Risk Level", 
                    "datapoints": [[system_risk_level, int(datetime.now().timestamp() * 1000)]]
                })
            
            elif target_name == "model_health_score":
                global model_accuracy
                health_score = min(100, model_accuracy + random.uniform(-5, 5))
                results.append({
                    "target": "Model Health Score",
                    "datapoints": [[health_score, int(datetime.now().timestamp() * 1000)]]
                })
            
            elif target_name == "model_status":
                results.append({
                    "target": "Model Status",
                    "datapoints": [[model_status, int(datetime.now().timestamp() * 1000)]]
                })
            
            elif target_name == "model_accuracy":
                results.append({
                    "target": "Model Accuracy",
                    "datapoints": [[model_accuracy, int(datetime.now().timestamp() * 1000)]]
                })
        
        logger.info(f"Returning {len(results)} result series")
        for result in results:
            datapoint_count = len(result.get('datapoints', []))
            logger.info(f"  - {result.get('target')}: {datapoint_count} datapoints")
        
        return results
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/annotations")
async def annotations(request: Request):
    """Handle annotation queries for Grafana"""
    try:
        body = await request.json()
        logger.info(f"Annotations request: {body}")
        
        # Return sample anomaly annotations
        annotations = []
        
        if current_anomalies > 3:
            now = datetime.now()
            annotations.append({
                "annotation": {
                    "name": "Anomaly Events",
                    "enabled": True,
                    "datasource": "AI-Engine"
                },
                "title": "High CPU Anomaly Detected",
                "time": int((now - timedelta(minutes=30)).timestamp() * 1000),
                "text": "AI detected unusual CPU usage pattern",
                "tags": ["anomaly", "cpu", "high-severity"]
            })
        
        return annotations
        
    except Exception as e:
        logger.error(f"Annotations error: {e}")
        return []

@app.get("/models/discover")
async def discover_available_models():
    """Discover all available trained models and their capabilities"""
    try:
        # Get prediction models status
        prediction_status = prediction_manager.get_prediction_status()
        
        # Get anomaly models status  
        anomaly_status = anomaly_manager.get_model_status()
        
        discovery_info = {
            "timestamp": datetime.now().isoformat(),
            "prediction_models": {
                "count": prediction_status.get("available_predictors", 0),
                "metrics": prediction_status.get("metrics", []),
                "details": prediction_status.get("model_details", {}),
                "grafana_targets": []
            },
            "anomaly_models": {
                "status": anomaly_status.get("status", "no_model_loaded"),
                "features_count": anomaly_status.get("features_count", 0),
                "training_date": anomaly_status.get("training_date"),
                "performance": anomaly_status.get("performance_metrics", {})
            },
            "grafana_integration": {
                "available_prediction_targets": [],
                "target_mapping": {}
            }
        }
        
        # Generate Grafana targets dynamically
        trained_metrics = prediction_status.get("metrics", [])
        target_mapping = {}
        
        for metric_name in trained_metrics:
            # Create semantic Grafana target names
            if any(cpu_term in metric_name.lower() for cpu_term in ['cpu', 'processor']):
                target = "prediction_cpu_util"
                target_mapping[target] = metric_name
                discovery_info["grafana_integration"]["available_prediction_targets"].append(target)
            elif any(mem_term in metric_name.lower() for mem_term in ['memory', 'ram', 'mem']):
                target = "prediction_memory_util"
                target_mapping[target] = metric_name
                discovery_info["grafana_integration"]["available_prediction_targets"].append(target)
            elif any(disk_term in metric_name.lower() for disk_term in ['disk', 'fs', 'filesystem']):
                target = "prediction_disk_util"
                target_mapping[target] = metric_name
                discovery_info["grafana_integration"]["available_prediction_targets"].append(target)
            # Add more mappings as needed
        
        discovery_info["grafana_integration"]["target_mapping"] = target_mapping
        discovery_info["prediction_models"]["grafana_targets"] = list(target_mapping.keys())
        
        return discovery_info
        
    except Exception as e:
        logger.error(f"Model discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_prediction_data(metric_type: str, time_range: Dict) -> List[List]:
    """Generate synthetic prediction data"""
    try:
        # Parse time range
        from_time = datetime.fromtimestamp(time_range.get("from", datetime.now().timestamp() - 3600))
        to_time = datetime.fromtimestamp(time_range.get("to", datetime.now().timestamp()))
        
        # Generate data points every 5 minutes
        datapoints = []
        current_time = from_time
        
        while current_time <= to_time:
            timestamp_ms = int(current_time.timestamp() * 1000)
            
            if metric_type == "cpu":
                # Simulate CPU prediction with some noise
                base_value = 30 + 20 * np.sin(current_time.hour * np.pi / 12)
                noise = random.uniform(-5, 5)
                value = max(0, min(100, base_value + noise))
            else:  # memory
                # Simulate memory prediction
                base_value = 45 + 15 * np.sin((current_time.hour + 2) * np.pi / 12)
                noise = random.uniform(-3, 3)
                value = max(0, min(100, base_value + noise))
            
            datapoints.append([value, timestamp_ms])
            current_time += timedelta(minutes=5)
        
        return datapoints
        
    except Exception as e:
        logger.error(f"Prediction generation error: {e}")
        return []

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.app_version,
        "target_host": settings.monitoring.target_host
    }

@app.get("/config")
async def get_configuration():
    """Get current configuration (without sensitive data)"""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "target_host": settings.monitoring.target_host,
        "ml_update_interval": settings.ai.ml_update_interval,  # Updated field name
        "key_metrics": settings.monitoring.key_metrics,
        "debug": settings.debug,
        "ml_storage_path": str(settings.ai.ml_storage_path),  # Added new field
        "anomaly_threshold": settings.ai.anomaly_confidence_threshold  # Added threshold
    }

@app.post("/data/collect")
async def trigger_data_collection(
    hours_back: int = 24,
    include_alerts: bool = True
):
    """Manually trigger data collection"""
    try:
        metrics_df, alerts_df = await data_manager.collect_training_data(
            hours_back=hours_back,
            include_alerts=include_alerts
        )
        
        return {
            "status": "success",
            "metrics_count": len(metrics_df),
            "alerts_count": len(alerts_df),
            "collection_time": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/test")
async def test_data_connections():
    """Test data source connections"""
    try:
        success = await test_data_collection()
        
        return {
            "status": "success" if success else "failed",
            "timestamp": datetime.now().isoformat(),
            "influxdb_url": str(settings.database.influxdb_url),
            "mysql_host": settings.database.mysql_host,
            "target_host": settings.monitoring.target_host
        }
    
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/influxdb")
async def debug_influxdb_data():
    """Debug endpoint to inspect InfluxDB data"""
    try:
        from data_collector import InfluxDBCollector
        
        async with InfluxDBCollector() as collector:
            # Check available hosts
            hosts_query = '''
            import "influxdata/influxdb/schema"
            
            from(bucket: "zabbix-metrics")
                |> range(start: -7d)
                |> group(columns: ["host"])
                |> distinct(column: "host")
                |> limit(n: 50)
            '''
            
            # Check available measurements
            measurements_query = '''
            import "influxdata/influxdb/schema"
            schema.measurements(bucket: "zabbix-metrics")
                |> limit(n: 50)
            '''
            
            # Get recent data sample
            sample_query = '''
            from(bucket: "zabbix-metrics")
                |> range(start: -7d)
                |> limit(n: 20)
            '''
            
            query_api = collector.client.query_api()
            
            hosts_result = await query_api.query_data_frame(hosts_query)
            measurements_result = await query_api.query_data_frame(measurements_query)
            sample_result = await query_api.query_data_frame(sample_query)
            
            return {
                "available_hosts": hosts_result.to_dict('records') if not hosts_result.empty else [],
                "available_measurements": measurements_result.to_dict('records') if not measurements_result.empty else [],
                "sample_data": sample_result.to_dict('records') if not sample_result.empty else [],
                "target_host_configured": settings.monitoring.target_host,
                "key_metrics_configured": settings.monitoring.key_metrics
            }
    
    except Exception as e:
        logger.error(f"Debug query failed: {e}")
        return {
            "error": str(e),
            "target_host_configured": settings.monitoring.target_host,
            "key_metrics_configured": settings.monitoring.key_metrics
        }

@app.get("/debug/mysql")
async def debug_mysql_data():
    """Debug endpoint to inspect MySQL/Zabbix data"""
    try:
        from data_collector import MySQLCollector
        
        async with MySQLCollector() as collector:
            # Check available hosts in Zabbix
            hosts_query = """
            SELECT hostid, host, name, status 
            FROM hosts 
            WHERE status IN (0, 1)
            ORDER BY host
            LIMIT 20
            """
            
            # Check recent alerts
            alerts_query = """
            SELECT COUNT(*) as alert_count, MAX(clock) as latest_alert
            FROM alerts
            WHERE clock >= UNIX_TIMESTAMP(DATE_SUB(NOW(), INTERVAL 7 DAY))
            """
            
            async with collector.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Get hosts
                    await cursor.execute(hosts_query)
                    hosts = await cursor.fetchall()
                    
                    # Get alerts summary
                    await cursor.execute(alerts_query)
                    alerts_summary = await cursor.fetchone()
            
            return {
                "available_hosts": [
                    {"hostid": h[0], "host": h[1], "name": h[2], "status": h[3]} 
                    for h in hosts
                ],
                "alerts_summary": {
                    "count": alerts_summary[0] if alerts_summary else 0,
                    "latest": alerts_summary[1] if alerts_summary else None
                },
                "target_host_configured": settings.monitoring.target_host
            }
    
    except Exception as e:
        logger.error(f"MySQL debug query failed: {e}")
        return {
            "error": str(e),
            "target_host_configured": settings.monitoring.target_host
        }

# UPDATED ANOMALY DETECTION ENDPOINTS

@app.post("/models/train")
async def train_anomaly_models(
    hours_back: int = 24,
    contamination: float = 0.1,
    hyperparameter_tuning: bool = True
):
    """Train anomaly detection models with robust data handling"""
    try:
        # Use the new fallback training method
        metadata = await anomaly_manager.train_models_with_fallback(
            data_manager,
            hours_back=hours_back,
            contamination=contamination,
            hyperparameter_tuning=hyperparameter_tuning
        )
        
        return {
            "status": "success",
            "model_version": metadata.version,
            "training_samples": metadata.training_samples,
            "features_count": metadata.features_count,
            "performance_metrics": metadata.performance_metrics,
            "training_date": metadata.training_date.isoformat(),
            "contamination_rate": metadata.contamination_rate,
            "message": f"Model trained successfully with {metadata.training_samples} samples"
        }
    
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/detect-anomalies")
async def predict_anomalies(
    hours_back: int = 1,
    return_explanations: bool = True,
    severity_filter: str = None  # NEW: Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)
):
    """Detect anomalies in recent data with robust preprocessing"""
    try:
        # Collect recent data
        metrics_df, _ = await data_manager.collect_training_data(
            hours_back=hours_back,
            include_alerts=False
        )
        
        if metrics_df.empty:
            raise HTTPException(status_code=400, detail="No recent data available")
        
        # Use simple feature preparation instead of complex ML pipeline
        feature_data = anomaly_manager._prepare_simple_features(metrics_df)
        
        if feature_data.empty:
            raise HTTPException(status_code=400, detail="No valid features after preprocessing")
        
        # Detect anomalies
        results = anomaly_manager.detect_anomalies_realtime(
            feature_data,
            return_explanations=return_explanations
        )
        
        # NEW: Filter by severity if requested
        if severity_filter and severity_filter.upper() in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            filtered_anomalies = [
                anomaly for anomaly in results["anomalies"] 
                if anomaly["severity"] == severity_filter.upper()
            ]
            results["anomalies"] = filtered_anomalies
            results["filtered_anomaly_count"] = len(filtered_anomalies)
            results["filter_applied"] = severity_filter.upper()
        
        return results
    
    except Exception as e:
        logger.error(f"Anomaly prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def get_model_status():
    """Get current model status and detailed information"""
    try:
        status = anomaly_manager.get_model_status()
        
        # Add additional useful information
        if status.get("status") == "model_loaded":
            status["model_health"] = "healthy"
            status["ready_for_detection"] = True
            
            # Calculate time since training
            from datetime import datetime
            training_date = datetime.fromisoformat(status["training_date"])
            time_since_training = datetime.now() - training_date
            status["hours_since_training"] = round(time_since_training.total_seconds() / 3600, 1)
            
        return status
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load")
async def load_model(model_path: str = None):
    """Load a specific model or the latest one"""
    try:
        success = anomaly_manager.load_model(model_path)
        
        if success:
            status = anomaly_manager.get_model_status()
            return {
                "status": "success",
                "message": "Model loaded successfully",
                "model_info": status
            }
        else:
            raise HTTPException(status_code=404, detail="Model not found or failed to load")
    
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW ENDPOINTS for better functionality:

@app.get("/models/test-simple")
async def test_simple_training():
    """Test simple training without complex feature engineering"""
    try:
        # Collect basic data
        metrics_df, _ = await data_manager.collect_training_data(hours_back=24)
        
        if metrics_df.empty:
            return {"status": "no_data", "message": "No data available"}
        
        # Use simple feature preparation
        features_df = anomaly_manager._prepare_simple_features(metrics_df)
        
        if features_df.empty:
            return {"status": "no_features", "message": "No features after preprocessing"}
        
        # Train with simple features
        metadata = anomaly_manager.train_models(
            features_df, 
            contamination=0.1,
            hyperparameter_tuning=False
        )
        
        return {
            "status": "success",
            "message": "Simple training completed successfully",
            "training_samples": metadata.training_samples,
            "features_count": metadata.features_count,
            "model_version": metadata.version,
            "performance_summary": {
                "anomaly_rate": f"{metadata.performance_metrics.get('anomaly_detection_rate', 0):.1%}",
                "ensemble_agreement": f"{metadata.performance_metrics.get('ensemble_agreement', 0):.1%}",
                "high_confidence_anomalies": metadata.performance_metrics.get('high_confidence_anomalies', 0)
            }
        }
    
    except Exception as e:
        logger.error(f"Simple training test failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/models/anomalies/summary")
async def get_anomaly_summary(hours_back: int = 1):
    """Get a summary of recent anomalies without detailed explanations"""
    try:
        # Quick anomaly check
        metrics_df, _ = await data_manager.collect_training_data(
            hours_back=hours_back,
            include_alerts=False
        )
        
        if metrics_df.empty:
            return {"status": "no_data", "total_samples": 0, "anomaly_count": 0}
        
        feature_data = anomaly_manager._prepare_simple_features(metrics_df)
        
        if feature_data.empty:
            return {"status": "no_features", "total_samples": 0, "anomaly_count": 0}
        
        # Get quick results without explanations
        results = anomaly_manager.detect_anomalies_realtime(
            feature_data,
            return_explanations=False
        )
        
        # Summarize by severity
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for anomaly in results["anomalies"]:
            severity_counts[anomaly["severity"]] += 1
        
        return {
            "status": "success",
            "timestamp": results["timestamp"],
            "total_samples": results["total_samples"],
            "anomaly_count": results["anomaly_count"],
            "anomaly_rate": results["anomaly_rate"],
            "processing_time_seconds": results["processing_time_seconds"],
            "severity_breakdown": severity_counts,
            "highest_confidence": max([a["confidence_score"] for a in results["anomalies"]], default=0),
            "has_critical_anomalies": severity_counts["CRITICAL"] > 0
        }
    
    except Exception as e:
        logger.error(f"Anomaly summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/metrics")
async def get_model_metrics():
    """Get detailed model performance metrics"""
    try:
        status = anomaly_manager.get_model_status()
        
        if status.get("status") != "model_loaded":
            raise HTTPException(status_code=400, detail="No model loaded")
        
        metrics = status.get("performance_metrics", {})
        
        return {
            "model_version": status["version"],
            "training_date": status["training_date"],
            "training_samples": status["training_samples"],
            "features_count": status["features_count"],
            "unsupervised_metrics": {
                "anomaly_detection_rate": f"{metrics.get('anomaly_detection_rate', 0):.2%}",
                "mean_anomaly_score": round(metrics.get('mean_anomaly_score', 0), 3),
                "ensemble_agreement": f"{metrics.get('ensemble_agreement', 0):.1%}",
                "silhouette_score": round(metrics.get('silhouette_score', 0), 3)
            },
            "anomaly_distribution": {
                "high_confidence": metrics.get('high_confidence_anomalies', 0),
                "medium_confidence": metrics.get('medium_confidence_anomalies', 0),
                "low_confidence": metrics.get('low_confidence_anomalies', 0),
                "total_detected": metrics.get('total_anomalies_detected', 0)
            },
            "model_health": "excellent" if metrics.get('ensemble_agreement', 0) > 0.85 else "good" if metrics.get('ensemble_agreement', 0) > 0.7 else "needs_attention"
        }
    
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/retrain")
async def retrain_model(
    hours_back: int = 48,
    force_retrain: bool = False
):
    """Retrain the model with more recent data"""
    try:
        status = anomaly_manager.get_model_status()
        
        # Check if retrain is needed
        if not force_retrain and status.get("status") == "model_loaded":
            from datetime import datetime
            training_date = datetime.fromisoformat(status["training_date"])
            hours_since_training = (datetime.now() - training_date).total_seconds() / 3600
            
            if hours_since_training < 24:  # Model is less than 24 hours old
                return {
                    "status": "skipped",
                    "message": f"Model is only {hours_since_training:.1f} hours old. Use force_retrain=true to override.",
                    "current_model_version": status["version"]
                }
        
        # Retrain with more data
        metadata = await anomaly_manager.train_models_with_fallback(
            data_manager,
            hours_back=hours_back,
            contamination=0.1,
            hyperparameter_tuning=True
        )
        
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "new_model_version": metadata.version,
            "training_samples": metadata.training_samples,
            "features_count": metadata.features_count,
            "improvement": {
                "previous_version": status.get("version", "none"),
                "new_performance": metadata.performance_metrics
            }
        }
    
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# PREDICTION ENGINE ENDPOINTS

@app.post("/predictions/train")
async def train_prediction_models(
    hours_back: int = 168,  # 1 week default
    forecast_horizon: int = 24,
    target_metrics: list[str] = None
):
    """Train prediction models for time series forecasting"""
    try:
        # Collect training data
        metrics_df, _ = await data_manager.collect_training_data(
            hours_back=hours_back,
            include_alerts=False
        )
        
        if metrics_df.empty:
            raise HTTPException(status_code=400, detail="No training data available")
        
        # Use clean metrics (not the heavily engineered ML dataset)
        clean_metrics = data_manager.preprocessor.clean_data(metrics_df)
        
        # Train prediction models
        results = prediction_manager.train_prediction_models(
            clean_metrics,
            target_metrics=target_metrics,
            forecast_horizon=forecast_horizon
        )
        
        if not results:
            raise HTTPException(status_code=400, detail="No prediction models were trained")
        
        # Format response
        response = {
            "status": "success",
            "trained_models": len(results),
            "forecast_horizon_hours": forecast_horizon,
            "models": {}
        }
        
        for metric, metadata in results.items():
            response["models"][metric] = {
                "model_type": metadata.model_type,
                "training_samples": metadata.training_samples,
                "performance_metrics": metadata.performance_metrics,
                "version": metadata.version
            }
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# Add this endpoint after your existing endpoints

@app.post("/predictions/train-now")
async def train_models_now():
    """Manually trigger model training (same as your curl command)"""
    try:
        logger.info("Manual training triggered...")
        
        target_metrics = [
            "system_cpu_util",
            "system_cpu_util_user", 
            "system_cpu_load_all_avg1",
            "vm_memory_utilization"
        ]
        
        # Collect training data (same parameters as your curl command)
        metrics_df, _ = await data_manager.collect_training_data(
            hours_back=168,  # 1 week
            include_alerts=False
        )
        
        if metrics_df.empty:
            raise HTTPException(status_code=400, detail="No training data available")
        
        # Use clean metrics
        clean_metrics = data_manager.preprocessor.clean_data(metrics_df)
        
        # Train prediction models (same as your curl command)
        results = prediction_manager.train_prediction_models(
            clean_metrics,
            target_metrics=target_metrics,
            forecast_horizon=24
        )
        
        if not results:
            raise HTTPException(status_code=400, detail="No prediction models were trained")
        
        # Format response (same as your original endpoint)
        response = {
            "status": "success",
            "trained_models": len(results),
            "forecast_horizon_hours": 24,
            "training_timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        for metric, metadata in results.items():
            response["models"][metric] = {
                "model_type": metadata.model_type,
                "training_samples": metadata.training_samples,
                "performance_metrics": metadata.performance_metrics,
                "version": metadata.version
            }
        
        logger.info(f"Manual training completed: {len(results)} models trained")
        return response
    
    except Exception as e:
        logger.error(f"Manual training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/auto-training/status")
async def get_auto_training_status():
    """Get auto-training status and next scheduled run"""
    try:
        status = prediction_manager.get_prediction_status()
        current_models = status.get("models", {})
        
        model_ages = {}
        oldest_model_hours = 0
        
        for metric, model_info in current_models.items():
            version = model_info.get("version", "")
            if version and "_" in version:
                try:
                    parts = version.split("_")
                    if len(parts) >= 3:
                        timestamp_str = f"{parts[-2]}_{parts[-1]}"
                        model_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        hours_old = (datetime.now() - model_time).total_seconds() / 3600
                        model_ages[metric] = round(hours_old, 1)
                        oldest_model_hours = max(oldest_model_hours, hours_old)
                except Exception:
                    model_ages[metric] = "unknown"
        
        return {
            "auto_training_enabled": True,
            "check_interval_hours": 0.5,
            "retrain_threshold_hours": 4,
            "current_models": len(current_models),
            "model_ages_hours": model_ages,
            "oldest_model_hours": round(oldest_model_hours, 1),
            "needs_retraining": oldest_model_hours > 4,
            "target_metrics": [
                "system_cpu_util",
                "system_cpu_util_user", 
                "system_cpu_load_all_avg1",
                "vm_memory_utilization"
            ]
        }
    
    except Exception as e:
        logger.error(f"Failed to get auto-training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/predictions/forecast")
async def generate_forecasts(
    forecast_horizon: int = 24,
    metrics: list[str] = None
):
    """Generate forecasts for the specified time horizon"""
    try:
        forecasts = prediction_manager.generate_forecasts(
            forecast_horizon=forecast_horizon,
            metrics=metrics
        )
        
        return {
            "status": "success",
            "forecast_data": forecasts
        }
    
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/status")
async def get_prediction_status():
    """Get status of all prediction models"""
    try:
        status = prediction_manager.get_prediction_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get prediction status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predictions/load")
async def load_prediction_models(metrics: list[str] = None):
    """Load trained prediction models"""
    try:
        success = prediction_manager.load_predictors(metrics)
        
        if success:
            status = prediction_manager.get_prediction_status()
            return {
                "status": "success",
                "message": "Prediction models loaded successfully",
                "model_info": status
            }
        else:
            raise HTTPException(status_code=404, detail="No prediction models found")
    
    except Exception as e:
        logger.error(f"Prediction model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add these endpoints to your main.py file after the existing ones

@app.get("/grafana/search")
async def grafana_search():
    """Grafana search endpoint for metrics discovery"""
    return [
        "anomalies_current",
        "anomalies_24h", 
        "prediction_cpu_util",
        "prediction_memory_util",
        "model_health_score",
        "system_risk_level",
        "forecast_alerts"
    ]

@app.post("/grafana/query")
async def grafana_query(request: dict):
    """Grafana query endpoint for time series data"""
    try:
        targets = request.get("targets", [])
        time_range = request.get("range", {})
        
        results = []
        
        for target in targets:
            target_metric = target.get("target", "")
            
            if target_metric == "anomalies_current":
                # Get current anomalies
                anomaly_data = await get_current_anomalies()
                results.append({
                    "target": "Current Anomalies",
                    "datapoints": anomaly_data
                })
                
            elif target_metric == "prediction_cpu_util":
                # Get CPU utilization predictions
                forecast_data = await get_metric_forecast("system_cpu_util", 24)
                results.append({
                    "target": "CPU Utilization Forecast",
                    "datapoints": forecast_data
                })
                
            elif target_metric == "model_health_score":
                # Get model health metrics
                health_data = await get_model_health_score()
                results.append({
                    "target": "Model Health Score",
                    "datapoints": health_data
                })
                
            elif target_metric == "system_risk_level":
                # Get system risk assessment
                risk_data = await get_system_risk_assessment()
                results.append({
                    "target": "System Risk Level",
                    "datapoints": risk_data
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Grafana query failed: {e}")
        return []

@app.get("/grafana/annotations")
async def grafana_annotations(request: dict):
    """Grafana annotations for anomaly events"""
    try:
        # Get recent anomalies for annotations
        annotations = await get_anomaly_annotations()
        return annotations
        
    except Exception as e:
        logger.error(f"Grafana annotations failed: {e}")
        return []

# Helper functions for Grafana integration
async def get_current_anomalies():
    """Get current anomaly metrics for Grafana"""
    try:
        # Get recent data and detect anomalies
        metrics_df, _ = await data_manager.collect_training_data(hours_back=1, include_alerts=False)
        
        if metrics_df.empty:
            return []
        
        # Use simple feature preparation
        feature_data = anomaly_manager._prepare_simple_features(metrics_df)
        
        if feature_data.empty:
            return []
        
        # Detect anomalies
        results = anomaly_manager.detect_anomalies_realtime(feature_data, return_explanations=False)
        
        # Convert to Grafana time series format
        datapoints = []
        current_time = datetime.now()
        
        # Create time series with anomaly count
        for i in range(60):  # Last 60 minutes
            timestamp = int((current_time - timedelta(minutes=i)).timestamp() * 1000)
            anomaly_count = results.get('anomaly_count', 0) if i == 0 else 0
            datapoints.append([anomaly_count, timestamp])
        
        return datapoints
        
    except Exception as e:
        logger.error(f"Error getting current anomalies: {e}")
        return []

async def get_metric_forecast(metric_name: str, hours: int):
    """Get metric forecast for Grafana"""
    try:
        # Generate forecasts
        forecasts = prediction_manager.generate_forecasts(
            forecast_horizon=hours,
            metrics=[metric_name]
        )
        
        datapoints = []
        if metric_name in forecasts.get('forecasts', {}):
            forecast_data = forecasts['forecasts'][metric_name]
            
            if 'prophet' in forecast_data.get('models', {}):
                prophet_data = forecast_data['models']['prophet']
                timestamps = prophet_data.get('timestamps', [])
                values = prophet_data.get('predicted_values', [])
                
                for timestamp, value in zip(timestamps, values):
                    ts = int(pd.to_datetime(timestamp).timestamp() * 1000)
                    datapoints.append([value, ts])
        
        return datapoints
        
    except Exception as e:
        logger.error(f"Error getting forecast for {metric_name}: {e}")
        return []

async def get_model_health_score():
    """Get model health score for Grafana"""
    try:
        status = anomaly_manager.get_model_status()
        
        # Calculate health score based on model status
        if status.get("status") == "model_loaded":
            performance = status.get("performance_metrics", {})
            ensemble_agreement = performance.get("ensemble_agreement", 0)
            health_score = ensemble_agreement * 100  # Convert to percentage
        else:
            health_score = 0
        
        # Create time series
        current_time = int(datetime.now().timestamp() * 1000)
        return [[health_score, current_time]]
        
    except Exception as e:
        logger.error(f"Error getting model health score: {e}")
        return [[0, int(datetime.now().timestamp() * 1000)]]

async def get_system_risk_assessment():
    """Get system risk level for Grafana"""
    try:
        # Get recent anomalies and calculate risk
        metrics_df, _ = await data_manager.collect_training_data(hours_back=1, include_alerts=False)
        
        if metrics_df.empty:
            risk_level = 0
        else:
            feature_data = anomaly_manager._prepare_simple_features(metrics_df)
            results = anomaly_manager.detect_anomalies_realtime(feature_data, return_explanations=False)
            
            # Calculate risk based on anomaly rate and severity
            anomaly_rate = results.get('anomaly_rate', 0)
            anomalies = results.get('anomalies', [])
            
            critical_count = sum(1 for a in anomalies if a.get('severity') == 'CRITICAL')
            high_count = sum(1 for a in anomalies if a.get('severity') == 'HIGH')
            
            # Risk calculation (0-100)
            risk_level = min(100, (anomaly_rate * 50) + (critical_count * 30) + (high_count * 15))
        
        current_time = int(datetime.now().timestamp() * 1000)
        return [[risk_level, current_time]]
        
    except Exception as e:
        logger.error(f"Error getting system risk: {e}")
        return [[0, int(datetime.now().timestamp() * 1000)]]

async def get_anomaly_annotations():
    """Get anomaly annotations for Grafana"""
    try:
        # Get recent anomalies
        metrics_df, _ = await data_manager.collect_training_data(hours_back=24, include_alerts=False)
        
        if metrics_df.empty:
            return []
        
        feature_data = anomaly_manager._prepare_simple_features(metrics_df)
        results = anomaly_manager.detect_anomalies_realtime(feature_data, return_explanations=True)
        
        annotations = []
        for anomaly in results.get('anomalies', []):
            timestamp = anomaly.get('timestamp')
            severity = anomaly.get('severity', 'LOW')
            confidence = anomaly.get('confidence_score', 0)
            
            if timestamp:
                ts = int(pd.to_datetime(timestamp).timestamp() * 1000)
                annotations.append({
                    "time": ts,
                    "title": f"{severity} Anomaly",
                    "text": f"Anomaly detected with {confidence:.2%} confidence",
                    "tags": [severity.lower(), "anomaly"]
                })
        
        return annotations
        
    except Exception as e:
        logger.error(f"Error getting anomaly annotations: {e}")
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




