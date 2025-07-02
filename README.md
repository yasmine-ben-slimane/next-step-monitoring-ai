# Zabbix AI-Powered Monitoring System

A comprehensive monitoring solution integrating Zabbix, InfluxDB, Grafana, and AI-powered anomaly detection and prediction engines.

##  Architecture Overview

This project provides an end-to-end monitoring ecosystem with:

- **Zabbix 6.0**: Core monitoring platform with MySQL backend
- **InfluxDB 2.7**: Time-series database for metrics storage
- **Grafana 10.2**: Advanced visualization and dashboards
- **AI Engine**: FastAPI-based machine learning service for anomaly detection and forecasting
- **Logstash**: Data pipeline for automated metric processing

##  Quick Start

### Prerequisites
- Docker 
- 8GB+ RAM recommended
- Available ports: 80, 3000, 3306, 8000, 8086, 10050, 10051

### Deployment

```bash
# Clone the repository
git clone https://github.com/yasmine-ben-slimane/next-step-monitoring-ai.git
cd next-step-monitoring-ai

# Start the entire stack
docker compose up -d

# Check service health
docker compose ps
```

### Initial Access

| Service | URL | Credentials |
|---------|-----|-------------|
| Zabbix Web | http://localhost | Admin/zabbix |
| Grafana | http://localhost:3000 | admin/admin123 |
| InfluxDB | http://localhost:8086 | admin/password123 |
| AI Engine API | http://localhost:8000 | No auth |

##  Components

### 1. Zabbix Monitoring Stack

- **MySQL 8.0**: Primary database with optimized configuration
- **Zabbix Server**: Core monitoring engine
- **Zabbix Web**: User interface (Apache + PHP 8.1)
- **Zabbix Agent**: System metrics collection

**Auto-configured host**: `ubuntu-2204-host` with system monitoring templates

### 2. Data Pipeline

```
Zabbix MySQL → Logstash → InfluxDB → Grafana + AI Engine
```

- **Automated Export**: 60-second intervals
- **45+ Key Metrics**: CPU, memory, disk, network, processes
- **Real-time Processing**: Logstash transforms and forwards data

### 3. AI Engine Features

#### Anomaly Detection
```bash
# Train anomaly detection models
curl -X POST "http://localhost:8000/models/train" \
  -H "Content-Type: application/json" \
  -d '{"hours_back": 24, "contamination": 0.1}'

# Detect real-time anomalies
curl -X POST "http://localhost:8000/models/detect-anomalies" \
  -H "Content-Type: application/json" \
  -d '{"hours_back": 1}'
```

#### Predictive Forecasting Example
```bash
# Train prediction models
curl -X POST "http://localhost:8000/predictions/train?hours_back=168&forecast_horizon=24" \
-H "Content-Type: application/json" \
-d '["system_cpu_util", "system_cpu_util_user", "system_cpu_load_all_avg1", "vm_memory_utilization"]'

# Generate 24-hour forecasts
curl -X POST "http://localhost:8000/predictions/forecast?forecast_horizon=24" \
-H "Content-Type: application/json" \
-d '["vm_memory_utilization"]'
```

### 4. Grafana Dashboards

#### Pre-configured Dashboards:

1. **System Monitoring** (`zabbix-system-monitoring.json`)
   - Real-time system metrics
   - CPU, memory, disk utilization
   - 30-second refresh rate

2. **AI Prediction & Forecast** (`prediction-forecast.json`)
   - 24-hour CPU/memory forecasts
   - Prophet/ARIMA model predictions
   - Confidence intervals

3. **Anomaly Detection** (`anomaly-detection.json`)
   - Real-time anomaly alerts
   - System risk assessment
   - Anomaly event annotations

4. **Model Health** (`model-health.json`)
   - AI model performance metrics
   - Training status and accuracy
   - Model lifecycle monitoring

##  AI Engine Architecture

### Configuration (`config.py`)
```python
# Database connections
INFLUXDB_URL = "http://influxdb:8086"
MYSQL_HOST = "mysql-server"

# AI settings
ML_UPDATE_INTERVAL = 3600  # 1 hour
ANOMALY_CONFIDENCE_THRESHOLD = 0.7
```

### Anomaly Detection (`anomaly_detection.py`)
- **Ensemble Methods**: Isolation Forest + LOF + Statistical
- **Real-time Processing**: Streaming anomaly detection
- **Explainable AI**: Feature contribution analysis
- **Severity Classification**: CRITICAL/HIGH/MEDIUM/LOW

### Prediction Engine (`prediction_engine.py`)
- **Time Series Models**: Prophet + ARIMA
- **Metric Profiles**: CPU, memory, network optimizations
- **24-hour Forecasts**: Confidence intervals included
- **Business Hour Seasonality**: Workday pattern recognition

### Data Collection (`data_collector.py`)
- **Async Processing**: High-performance data retrieval
- **Quality Assessment**: Data completeness scoring
- **Feature Engineering**: Automated ML feature creation

##  Monitoring Metrics

### Core System Metrics (45+ tracked):
- `vm_memory_utilization` - Memory usage percentage
- `system_cpu_util_user` - User CPU utilization
- `system_cpu_load_all_avg1` - 1-minute load average
- `proc_num` - Total process count
- `proc_num_run` - Running processes
- `system_uptime` - System uptime
- `system_cpu_intr` - CPU interrupts
- `agent_ping` - Agent connectivity
- `net_if_out_eth0` - Network output traffic
- `vfs_fs_size_rootfs_pused` - Root filesystem usage
- And 35+ more metrics...

- Side Note: in this project only the top 13 metrics are used, the rest are commented, feel free to uncomment them in the file ai-engine/config.py

### AI-Generated Metrics:
- Anomaly confidence scores
- System risk levels
- Prediction accuracy
- Model health indicators

##  Configuration

### Environment Variables
```yaml
# InfluxDB
INFLUXDB_TOKEN: "my-super-secret-auth-token"
INFLUXDB_ORG: "zabbix-org"
INFLUXDB_BUCKET: "zabbix-metrics"

# MySQL
MYSQL_USER: "zabbix"
MYSQL_PASSWORD: "root"
MYSQL_DATABASE: "zabbix"

# Timezone
TZ: "Africa/Tunis"
```

### Auto-Training & Automation
The AI Engine includes automatic model management:
- **Auto-retraining**: Every 30 minutes (if models > 4 hours old)
- **Auto-forecasting**: Continuous 24-hour predictions
- **Health monitoring**: Model performance tracking

##  API Endpoints

### Health & Status
```bash
GET /health                    # Service health check
GET /config                    # Configuration info
GET /models/status            # AI model status
```

### Data Collection
```bash
POST /data/collect            # Manual data collection
GET /data/test               # Test data connections
GET /debug/influxdb          # InfluxDB debug info
```

### Anomaly Detection
```bash
POST /models/train           # Train anomaly models
POST /models/detect-anomalies # Real-time detection
GET /models/anomalies/summary # Quick anomaly overview
POST /models/retrain         # Retrain with new data
GET /models/metrics          # Model performance metrics
```

### Predictions
```bash
POST /predictions/train      # Train forecast models
POST /predictions/train-now  # Manual training trigger
POST /predictions/forecast   # Generate forecasts
GET /predictions/status      # Model status
GET /predictions/auto-training/status # Auto-training status
```

### Grafana Integration
```bash
POST /search                 # Available metrics
POST /query                  # Time series data
POST /annotations           # Anomaly annotations
GET /models/discover        # Available model discovery
```

##  Troubleshooting

### Common Issues

1. **Service Health Checks**
```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs zabbix-server
docker-compose logs ai-engine
```

2. **Database Connections**
```bash
# Test InfluxDB
curl http://localhost:8086/ping

# Test AI Engine
curl http://localhost:8000/health
```

3. **Data Flow Verification**
```bash
# Check if metrics are flowing
curl "http://localhost:8000/debug/influxdb"

# Verify Zabbix agent
docker-compose logs zabbix-agent
```

### Performance Tuning

- **Memory**: Increase Docker memory allocation for ML workloads
- **Storage**: Use SSD for InfluxDB and MySQL volumes
- **Network**: Ensure low latency between services

##  Development

### Adding Custom Metrics
1. Update `config.py` monitoring key_metrics
2. Modify Logstash pipeline configuration
3. Retrain AI models with new metrics

### Custom Dashboards
1. Create JSON dashboard files in `grafana/dashboards/`
2. Use provisioning to auto-import
3. Configure data sources in `grafana/provisioning/`

##  Security Considerations

- Change default passwords in production
- Use environment files for sensitive data
- Enable TLS for external access
- Implement authentication for AI Engine API

##  Performance Metrics

### Expected Performance:
- **Data Collection**: ~60-second intervals
- **Anomaly Detection**: <2 seconds response time
- **Forecast Generation**: <30 seconds for 24h predictions
- **Dashboard Refresh**: 30-second intervals

### Resource Usage:
- **Memory**: 6-8GB total stack
- **CPU**: 2-4 cores recommended
- **Storage**: 10GB+ for time-series data
- **Network**: Minimal bandwidth requirements

##  AI Model Features

### Anomaly Detection Models
- **Ensemble Approach**: Multiple algorithms working together
- **Statistical Methods**: Z-score, Modified Z-score, IQR
- **Machine Learning**: Isolation Forest, Local Outlier Factor
- **Real-time Scoring**: Confidence-based severity assessment

### Prediction Models
- **Prophet**: Facebook's time series forecasting
- **ARIMA**: Auto-regressive integrated moving average
- **Metric Profiles**: Optimized parameters per metric type
- **Confidence Intervals**: Upper/lower bounds for predictions

### Auto-Learning Features
- **Continuous Training**: Models retrain automatically
- **Data Quality Assessment**: Monitors input data health
- **Performance Tracking**: Accuracy metrics and alerting
- **Fallback Strategies**: Robust error handling

##  Use Cases

1. **Proactive Monitoring**: Predict system issues before they occur
2. **Anomaly Detection**: Automatically identify unusual system behavior
3. **Capacity Planning**: Forecast resource requirements
4. **Performance Optimization**: Identify bottlenecks and trends
5. **Compliance**: Maintain audit trails and performance records
6. **Alert Reduction**: Smart filtering to reduce false positives
7. **Root Cause Analysis**: Correlate anomalies with system events

##  Documentation

### AI Engine Components

- **Enhanced Time Series Forecaster**: Metric-specific optimization
- **Ensemble Anomaly Detector**: Multi-algorithm approach
- **Feature Engineering Pipeline**: Automated feature creation
- **Data Quality Assessment**: Input validation and scoring
- **Model Metadata Management**: Version control and tracking

This comprehensive monitoring solution provides enterprise-grade observability with AI-powered insights for modern infrastructure management.
