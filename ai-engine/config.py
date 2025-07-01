from pydantic_settings import BaseSettings
from pydantic import validator, AnyHttpUrl
from typing import List, Optional
from pathlib import Path
import os

class DatabaseConfig(BaseSettings):
    """Database connection configuration"""
    
    # InfluxDB Configuration
    influxdb_url: AnyHttpUrl = "http://influxdb:8086"
    influxdb_token: str = "my-super-secret-auth-token"
    influxdb_org: str = "zabbix-org"
    influxdb_bucket: str = "zabbix-metrics"
    
    # MySQL Configuration
    mysql_host: str = "mysql-server"
    mysql_user: str = "zabbix"
    mysql_password: str = "root"
    mysql_database: str = "zabbix"
    mysql_port: int = 3306
    
    @validator('mysql_port')
    def validate_mysql_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('MySQL port must be between 1 and 65535')
        return v

class AIConfig(BaseSettings):
    """AI/ML specific configuration"""
    
    # Model Training & Updates (renamed to avoid conflicts)
    ml_update_interval: int = 3600  # 1 hour in seconds
    anomaly_confidence_threshold: float = 0.7
    prediction_timeframes: List[int] = [900, 1800, 3600]  # 15min, 30min, 1hour
    batch_processing_size: int = 1000
    
    # Model Storage (renamed to avoid conflicts)
    ml_storage_path: Path = Path("/app/models")
    ml_backup_count: int = 5
    
    # Feature Engineering
    rolling_window_sizes: List[int] = [5, 15, 60]  # minutes
    feature_lag_periods: List[int] = [1, 2, 3, 6, 12]  # periods
    
    # Data Quality
    outlier_threshold: float = 3.0  # standard deviations
    missing_data_threshold: float = 0.1  # 10% maximum missing data
    
    # Pydantic model configuration
    model_config = {"protected_namespaces": ()}
    
    @validator('anomaly_confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v
    
    @validator('ml_update_interval')
    def validate_update_interval(cls, v):
        if v < 300:  # minimum 5 minutes
            raise ValueError('Model update interval must be at least 300 seconds')
        return v
    
    @validator('batch_processing_size')
    def validate_batch_size(cls, v):
        if v < 1:
            raise ValueError('Batch processing size must be positive')
        return v

class LoggingConfig(BaseSettings):
    """Logging configuration"""
    
    log_level: str = "INFO"
    log_file_path: Path = Path("/app/logs/ai_engine.log")
    log_rotation: str = "1 day"
    log_retention: str = "7 days"
    log_format: str = "{time} | {level} | {name} | {message}"
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()

class MonitoringConfig(BaseSettings):
    """System monitoring configuration"""
    
    target_host: str = "ubuntu-2204-host"
    health_check_interval: int = 30
    metrics_collection_interval: int = 60
    alert_notification_url: Optional[str] = None
    
    # Key metrics to monitor (based on your InfluxDB data)
    key_metrics: List[str] = [
        # Top tier - Most frequent and important
        "vm_memory_utilization",
        "system_cpu_util_user", 
        "proc_num",
        "proc_num_run",
        "system_uptime",
        "system_cpu_load_all_avg1",
        "system_cpu_load_all_avg5", 
        "system_cpu_load_all_avg15",
        "system_cpu_intr",
        "agent_ping",
        "system_cpu_util_system",
        "net_if_out_eth0",
        "net_if_out_eth0_errors"
        #,
        
        # Second tier - Good data volume
        #"system_cpu_switches",
        #"system_localtime", 
        #"system_cpu_util_guest",
        #"system_users_num",
        #"system_cpu_util_softirq",
        #"system_cpu_util_iowait",
        #"system_cpu_util_nice",
        #"system_cpu_util_steal",
        #"system_swap_size_free",
        #"system_swap_size_total",
        #"system_cpu_util_idle",
        #"system_cpu_util_interrupt",
        #"system_cpu_util",
        
        # Network metrics
        #"net_if_in_eth0",
        #"net_if_in_eth0_errors", 
        #"net_if_in_eth0_dropped",
        #"net_if_out_eth0_dropped",
        
        # Filesystem (most critical ones)
        #"vfs_fs_size_rootfs_pused",
        #"vfs_fs_size_var_log_pused"
    ]

class Settings(BaseSettings):
    """Main application settings"""
    
    # Application Info
    app_name: str = "Zabbix AI Engine"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Configuration sections
    database: DatabaseConfig = DatabaseConfig()
    ai: AIConfig = AIConfig()
    logging: LoggingConfig = LoggingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"
        
    def get_mysql_connection_string(self) -> str:
        """Generate MySQL connection string"""
        return (f"mysql://{self.database.mysql_user}:{self.database.mysql_password}"
                f"@{self.database.mysql_host}:{self.database.mysql_port}"
                f"/{self.database.mysql_database}")
    
    def get_influxdb_connection_params(self) -> dict:
        """Get InfluxDB connection parameters"""
        return {
            "url": str(self.database.influxdb_url),
            "token": self.database.influxdb_token,
            "org": self.database.influxdb_org,
            "bucket": self.database.influxdb_bucket
        }

# Global settings instance
settings = Settings()

# Environment-specific configurations
def get_development_settings() -> Settings:
    """Development environment settings"""
    dev_settings = Settings()
    dev_settings.debug = True
    dev_settings.logging.log_level = "DEBUG"
    dev_settings.ai.ml_update_interval = 1800  # 30 minutes for faster development
    return dev_settings

def get_production_settings() -> Settings:
    """Production environment settings"""
    prod_settings = Settings()
    prod_settings.debug = False
    prod_settings.logging.log_level = "INFO"
    prod_settings.ai.ml_update_interval = 3600  # 1 hour
    return prod_settings

# Configuration validation
def validate_configuration() -> bool:
    """Validate all configuration settings"""
    try:
        # Test database connections
        settings.get_mysql_connection_string()
        settings.get_influxdb_connection_params()
        
        # Validate paths exist or can be created
        settings.ai.ml_storage_path.mkdir(parents=True, exist_ok=True)
        settings.logging.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False
