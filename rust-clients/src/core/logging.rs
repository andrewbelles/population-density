///
/// logging.rs  Andrew Belles  Dec 3rd, 2025 
///
/// Implementation of per client logging tool to 
/// provide informative logs as Client collects information 
///
///

use std::str::FromStr; 

use thiserror::Error; 
use tracing_subscriber::{fmt, Layer, layer::SubscriberExt, EnvFilter, Registry}; 

use crate::core::config::{LogLevel, LoggingConfig}; 

#[derive(Debug, Error)]
pub enum LoggingError {
    #[error("invalid log level: {0}")]
    Level(String), 
    #[error("otlp setup failed: {0}")]
    Otlp(String)
}

fn level_to_filter(level: &LogLevel) -> Result<EnvFilter, LoggingError> {
    let lvl_str = match level {
        LogLevel::Error => "error", 
        LogLevel::Warn  => "warn", 
        LogLevel::Info  => "info",
        LogLevel::Debug => "debug",
        LogLevel::Trace => "trace"
    }; 
    EnvFilter::from_str(lvl_str).map_err(|_| LoggingError::Level(lvl_str.into()))
}

pub fn init_logging(config: &LoggingConfig) -> Result<(), LoggingError> {
    let mut filter = level_to_filter(&config.log_level)?; 
    if let Ok(env) = std::env::var("RUST_LOG") {
        if let Ok(env_filter) = EnvFilter::try_new(env) {
            filter = env_filter; 
        }
    }

    let fmt_layer = if config.json {
        fmt::layer().event_format(fmt::format().json()).with_target(false).boxed() 
    } else {
        fmt::layer().with_target(true).boxed() 
    };

    let subscriber = Registry::default().with(filter).with(fmt_layer); 

    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| LoggingError::Otlp(e.to_string()))
}
