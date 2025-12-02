/// 
/// config.rs  Andrew Belles  Dec 2nd, 2025 
///
/// Details configuration structures for API Client Base 
/// 
/// Client is thread safe assuming configuration provided is valid 
///

use std::collections::HashMap; 
use std::path::PathBuf; 
use std::time::Duration; 

use serde::{Serialize, Deserialize};
use url::Url; 

/************ ClientConfig ********************************/ 
/* Parent Configuration fully detailing behavior of a single Client
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    pub http: HttpConfig, 
    pub storage: StorageConfig, 
    pub log: LoggingConfig, 
    #[serde(default)]
    pub runner: RunnerConfig, 
    #[serde(default)]
    pub endpoint_overrides: HashMap<String, EndpointConfig> 
}

/************ EndpointConfig ******************************/ 
/* Configuration for a single endpoint that a Client hits */ 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    pub path: String, 
    #[serde(default)]
    pub query_defaults: HashMap<String, String>, 
    #[serde(default)]
    pub pagination: Option<PaginationConfig>, 
    #[serde(default)]
    pub retry: Option<RetryConfig>, 
    #[serde(default)]
    pub rate_limit: Option<RateLimitConfig>
}

/************ HttpConfig **********************************/ 
/* Stores default information passed over https to targetted API
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    pub base_url: Url, 
    pub api_key: ApiKey, 
    #[serde(default = "default_user_agent")]
    pub user_agent: String, 
    #[serde(default = "default_timeout")]
    pub timeout: Duration,
    #[serde(default = "default_max_redirects")]
    pub max_redirects: u8,
    #[serde(default)]
    pub retry: RetryConfig, 
    #[serde(default)]
    pub default_headers: HashMap<String, String> 
}

/************ ApiKey **************************************/ 
/* Explicit binding over String */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey(pub String);

/************ RetryConfig *********************************/ 
/* Configurable information for client retrying a single endpoint 
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    #[serde(default = "default_max_retries")]
    pub max_retries: u32, 
    #[serde(default)]
    pub backoff: BackoffConfig, 
    #[serde(default)]
    pub retryable_statuses: Vec<u16>,
    #[serde(default)]
    pub retryable_errors: Vec<String>, 
}

/************ BackoffConfig *******************************/ 
/* Configurable information for client waiting before retry 
 */ 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackoffConfig {
    #[serde(default = "default_backoff_base")]
    pub base: Duration, 
    #[serde(default = "default_backoff_max")]
    pub max: Duration, 
    #[serde(default = "default_backoff_multiplier")]
    pub multiplier: f32 
} 

/************ Pagination Supporting ***********************/ 

/************ PaginationFSM *******************************/
/* Possible States requiring action by Pagination logic 
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaginationFSM {
    OffsetLimit,
    PageNumber, 
    Cursor
}

/************ PaginationParams ****************************/ 
/* Parameters required to fully determine strategy for each 
 * State Pagination logic needs to handle  
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_page_param")]
    pub page: String, 
    #[serde(default = "default_per_page_param")]
    pub per_page: String, 
    #[serde(default = "default_offset_param")]
    pub offset: String, 
    #[serde(default = "default_cursor_param")]
    pub cursor: String, 
    #[serde(default = "default_has_more_field")]
    pub has_more_field: String, 
    #[serde(default = "default_next_cursor_field")]
    pub next_cursor_field: String
}

/************ PaginationConfig ****************************/ 
/* Configurable information for client paginating incoming 
 * JSON received from API.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationConfig {
    pub strategy: PaginationFSM, 
    #[serde(default)]
    pub page_size: Option<u32>, 
    #[serde(default)]
    pub max_pages: Option<u32>, 
    #[serde(default)]
    pub param_names: PaginationParams, 
    #[serde(default)]
    pub initial_cursor: Option<String>, 
    #[serde(default = "default_true")]
    pub stop_on_duplicate: bool 
}

/************ Storage Supporting **************************/ 

/************ MetricsConfig *******************************/ 
/* Extra configuration for logging information related to storage 
 * of pulled information. Required by StorageConfig and LoggingConfig 
 */ 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool, 
    #[serde(default = "default_export_interval")]
    pub export_interval: Duration, 
    #[serde(default = "default_metrics_prefix")]
    pub prefix: String 
}

/************ StorageConfig *******************************/ 
/* Configuration of how Client interacts with its respective 
 * database correctly 
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub db_path: PathBuf, 
    #[serde(default)]
    pub journal_mode: Option<String>,  
    #[serde(default)]
    pub busy_timeout_ms: Option<u64>, 
    #[serde(default)]
    pub max_connections: Option<u32>,
    #[serde(default)]
    pub metrics: MetricsConfig 
}

/************ Logging Supporting **************************/ 

/************ LogLevel ************************************/ 
/* Logger State Enum */ 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error, 
    Warn, 
    Info, 
    Debug, 
    Trace 
}

/************ LoggingConfig *******************************/ 
/* Configuration for supporting logger 
 */ 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub log_level: LogLevel,
    #[serde(default)]
    pub json: bool, 
    #[serde(default)]
    pub otlp_endpoint: Option<String>, // open telemetry protocol used by tracing 
    #[serde(default)]
    pub metrics: MetricsConfig 
}

/************ RunnerConfig ********************************/ 
/* Configuration for async runner backbone that clients require 
 * to make them thread safe and allow async collection of data 
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerConfig {
    #[serde(default = "default_concurrency")]
    pub concurrency: usize, 
    #[serde(default = "default_queue_bound")]
    pub queue_bound: usize, 
    #[serde(default)]
    pub idle_shutdown_secs: Option<u64>, 
    #[serde(default = "default_graceful_shutdown_secs")]
    pub graceful_shutdown_secs: u64 
}

/************ RateLimitConfig *****************************/ 
/* Configures rate limiting of client */ 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32, 
    #[serde(default)]
    pub burst: u32, 
    #[serde(default)]
    pub cooldown: Option<Duration> 
}

/************ Defaults ************************************/ 
fn default_user_agent() -> String { "topographic-client/0.1".into() }
fn default_timeout() -> Duration { Duration::from_secs(30) }
fn default_max_redirects() -> u8 { 5 }
fn default_max_retries() -> u32 { 3 }
fn default_backoff_base() -> Duration { Duration::from_millis(200) }
fn default_backoff_max() -> Duration { Duration::from_secs(30) }
fn default_backoff_multiplier() -> f32 { 2.0 }
fn default_true() -> bool { true }
fn default_page_param() -> String { "page".into() }
fn default_per_page_param() -> String { "per_page".into() }
fn default_offset_param() -> String { "offset".into() }
fn default_cursor_param() -> String { "cursor".into() }
fn default_has_more_field() -> String { "has_more".into() }
fn default_next_cursor_field() -> String { "next_cursor".into() }
fn default_log_level() -> LogLevel { LogLevel::Info }
fn default_export_interval() -> Duration { Duration::from_secs(15) }
fn default_metrics_prefix() -> String { "topographic".into() }
fn default_concurrency() -> usize { 4 }
fn default_queue_bound() -> usize { 100 }
fn default_graceful_shutdown_secs() -> u64 { 10 }
