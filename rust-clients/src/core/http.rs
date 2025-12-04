/// 
/// http.rs  Andrew Belles  Dec 2nd, 2025 
///
/// Construction of HttpClient and methods for 
/// safe requests on API. 
///
/// AUTHORIZATION assumes format "Bearer {}" 
/// Only works for cloneable requests, need separate impl for streaming bodies
///

// use rand::{thread_rng, Rng};
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT}, 
    Client,
    ClientBuilder,
    RequestBuilder,
    Response, 
    StatusCode
};
use serde::de::DeserializeOwned; 
use thiserror::Error; 
use tokio::time::sleep; 
use url::Url; 

use crate::core::config::{BackoffConfig, HttpConfig, RetryConfig}; 
use crate::core::endpoint::{Endpoint, Method}; 

/************ HttpError ***********************************/ 
#[derive(Debug, Error)]
pub enum HttpError {
    #[error("failed to build URL: {0}")]
    Url(#[from] url::ParseError),
    #[error("failed to clone request for retry")]
    UnclonableRequest,
    #[error("http status {status}: {body}")]
    Status { status: StatusCode, body: String }, 
    #[error("reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
}

/************ HttpClient **********************************/ 
#[derive(Clone)]
pub struct HttpClient {
    client: Client, 
    base_url: Url, 
    config: HttpConfig, // see config.rs  
}

/************ HttpClient Implementations ******************/ 
/* Implementations for safe requests to API with retry/backoff */
impl HttpClient {
    pub fn new(config: HttpConfig) -> Result<Self, HttpError> {
        let mut headers = HeaderMap::new(); 
        let bearer      = format!("Bearer {}", config.api_key.0); 
        
        // Get key and user agent into header 
        headers.insert(
            AUTHORIZATION, 
            HeaderValue::from_str(&bearer).expect("invalid api key header")
        );
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&config.user_agent).expect("invalid user agent")
        ); 

        // Insert all headers provided by config 
        for (k, v) in &config.default_headers {
            let key: reqwest::header::HeaderName = 
                k.parse().expect("invalid default header name"); 
            let val = HeaderValue::from_str(v).expect("invalid default header value");
            headers.insert(key, val);
        }

        let client = ClientBuilder::new()
            .default_headers(headers)
            .timeout(config.timeout)
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects as usize))
            .build()?; 

        Ok(Self{ client, base_url: config.base_url.clone(), config })
    }

    pub async fn get_json<T>(
        &self, path: &str, query: &[(&str, String)]) -> Result<T, HttpError>
    where 
        T: DeserializeOwned
    {
        let url  = self.base_url.join(path)?; 
        let req  = self.client.get(url).query(query);
        let resp = self.execute_with_retry(req, None).await?; 
        Ok(resp.json().await?)
    }

    pub async fn post_json<T, B>(&self, path: &str, body: &B) -> Result<T, HttpError>
    where 
        T: DeserializeOwned,
        B: serde::Serialize + ?Sized, 
    {
        let url  = self.base_url.join(path)?; 
        let req  = self.client.post(url).json(body);
        let resp = self.execute_with_retry(req, None).await?; 
        Ok(resp.json().await?)
    }

    async fn execute_with_retry(&self, req: RequestBuilder, retry_override: Option<&RetryConfig>) 
        -> Result<Response, HttpError> {
        let retry = retry_override.unwrap_or(&self.config.retry); 
        let mut attempt: u32 = 0; 

        loop {
            let builder = req.try_clone().ok_or(HttpError::UnclonableRequest)?; 
            
            // Match return status with correct state action 
            match builder.send().await {
                // Successes/Retryables 
                Ok(resp) if resp.status().is_success() => return Ok(resp),
                Ok(resp) if (self.should_retry_status(resp.status()) && 
                        attempt < retry.max_retries) => {
                    self.backoff_sleep(attempt, &retry.backoff).await; 
                }

                // Errors
                Ok(resp) => {
                    let status = resp.status(); 
                    let body   = resp.text().await.unwrap_or_default();
                    return Err(HttpError::Status { status, body })
                }
                
                // Most likely error, possible retry 
                Err(err) if (self.is_retryable_error(&err, retry) &&
                        attempt < retry.max_retries) => {
                    self.backoff_sleep(attempt, &retry.backoff).await; 
                }

                // Error 
                Err(err) => return Err(HttpError::Reqwest(err)), 
            }

            attempt += 1;
        }
    }

    fn should_retry_status(&self, status: StatusCode) -> bool {
        if status.is_server_error() {
            return true; 
        }

        // Check Vec<u16> of retryable status from configurati on against Code 
        self.config.retry.retryable_statuses.iter()
            .any(|s| StatusCode::from_u16(*s).map_or(false, |code| code == status))
    }

    fn is_retryable_error(&self, err: &reqwest::Error, retry: &RetryConfig) -> bool {
        if err.is_timeout() || err.is_request() || err.is_connect() {
            return true; 
        }

        // Check if Msg contains any keywords from Vec<String> denoting we can retry 
        let msg = err.to_string(); 
        retry.retryable_errors.iter().any(|s| msg.contains(s))
    }

    async fn backoff_sleep(&self, attempt: u32, backoff: &BackoffConfig) {
        // Exponential backoff 
        let pow = backoff.multiplier.powi(attempt as i32);
        let mut delay = backoff.base.mul_f32(pow); 
        if delay > backoff.max {
            delay = backoff.max; 
        }

        sleep(delay).await; 
    }

    pub async fn request_endpoint<T, B>(&self, endpoint: &Endpoint<B>) -> Result<T, HttpError> 
    where 
        T: DeserializeOwned,
        B: serde::Serialize
    {
        let url = self.base_url.join(&endpoint.path)?; 
        let req = match endpoint.method {
            Method::Get  => self.client.get(url).query(&endpoint.query),
            Method::Post => {
                let mut builder = self.client.post(url).query(&endpoint.query);
                if let Some(body) = endpoint.body.as_ref() {
                    builder = builder.json(body); 
                }
                builder 
            } 
        }; 

        let resp = self.execute_with_retry(req, endpoint.retry.as_ref()).await?; 
        Ok(resp.json().await?)
    }
}
