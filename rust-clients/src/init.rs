///
/// init.rs  Andrew Belles  Dec 3rd, 2025
///
/// Instantiates Config for a single Client from File
///
/// Uses a client's name.yaml to configure the Client 
/// 

use std::{collections::HashMap, fs, path::Path, time::Duration}; 

use serde::Deserialize; 
use thiserror::Error;
use url::Url; 

use crate::core::config::{
    ApiKey, ClientConfig, EndpointConfig, HttpConfig, LoggingConfig, PaginationConfig,
    PaginationFSM, PaginationParams, RateLimitConfig, RunnerConfig, StorageConfig, MetricsConfig,
    RetryConfig 
}; 

/************ Configuration Load Errors *******************/

#[derive(Debug, Clone)] 
pub enum ConfigLoadError {
    #[error("io: {0}")]
    IO(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error), 
    #[error("url parse: {0}")]
    Url(#[from] url::ParseError), 
    #[error("missing client '{0}' in config")]
    MissingClient(String),
    #[error("missing api key env var '{0}'")]
    MissingApiKey(String)
}

/************ RawClient ***********************************/ 
/* Same as ClientConfig but wraps over RawHttp which is necessary 
 * as we must load the Api Key from environment so we cannot 
 * directly instantiate immediately 
 */ 
#[derive(Debug, Deserialize)] 
struct RawClient {
    http: RawHttp, 
    storage: StorageConfig, 
    log: LoggingConfig, 
    #[serde(default)]
    runner: RunnerConfig, 
    endpoints: HashMap<String, EndpointConfig> 
}

/************ RawHttp *************************************/ 
/* See RawClient for explanation. */ 
#[derive(Debug, Deserialize)] 
struct RawHttp {
    base_url: String, 
    api_key_env: String, 
    #[serde(default = "default_user_agent")]
    user_agent: String, 
    #[serde(default = "default_timeout_secs")]
    timeout_secs: u64, 
    #[serde(default = "default_max_redirects")]
    max_redirects: u8, 
    #[serde(default)]
    retry: RetryConfig, 
    #[serde(default)]
    default_headers: HashMap<String, String> 
}

/************ Default Helpers *****************************/ 
fn default_user_agent() -> String { "topographic-client/0.1".into() }
fn default_timeout_secs() -> u64  { 30 }
fn default_max_redirects() -> u8  { 5 }

type ClientConfigFile = HashMap<String, RawClient>; 

/************ load_client_from_yaml() *********************/ 
/* Loads new ClientConfig from YAML 
 *
 * Caller Provides: 
 *   path to config.yaml 
 *   name of client to pull from config  
 */
pub fn load_client_from_yaml(path: impl AsRef<Path>, client_name: &str) 
    -> Result<ClientConfig, ConfigLoadError> {
    let raw = fs::read_to_string(path)?; 
    let file: ClientConfigFile = serde_yaml::from_str(&raw)?; 
    let raw_client = file.get(client_name)
        .ok_or_else(|| ConfigLoadError::MissingClient(client_name.to_string()))?; 

    let api_key = std::env::var(&raw_client.http.api_key_env)
        .map_err(|_| ConfigLoadError::MissingApiKey(raw_client.http.api_key_env.clone()))?; 

    let http = HttpConfig {
        base_url: Url::parse(&raw_client.http.base_url)?, 
        api_key: ApiKey(api_key), 
        user_agent: raw_client.http.user_agent.clone(), 
        timeout: Duration::from_secs(raw_client.http.timeout_secs),
        max_redirects: raw_client.http.max_redirects, 
        retry: raw_client.http.retry.clone(), 
        default_headers: raw_client.http.default_headers.clone() 
    }; 

    Ok(ClientConfig {
        http, 
        storage: raw_client.storage.clone(), 
        log: raw_client.log.clone(), 
        runner: raw_client.runner.clone(), 
        endpoint_overrides: raw_client.endpoints.clone(), 
    })
}

