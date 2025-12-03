///
/// endpoint.rs  Andrew Belles  Dec 3rd, 2025 
///
/// Details the Endpoint struct for which Clients formulate 
/// a request from.
///
/// Provides Constructors to allow for the instantiation of an Endpoint 
/// from an EndpointConfig. Thread safe 
///

use std::sync::Arc; 
use serde::Serialize; 
use crate::core::config::{EndpointConfig, RateLimitConfig, RetryConfig}; 

/************ Endpoint::Method ****************************/ 
#[derive(Clone, Debug)]
pub enum Method {
    Get, 
    Post 
}

/************ Endpoint ************************************/ 
/* Exposed struct for a single endpoint containing all 
 * information that the Client needs to request data as well 
 * as manage rate limits. 
 */
#[derive(Clone, Debug)]
pub struct Endpoint<Body = serde_json::Value> {
    pub method: Method, 
    pub path: String, 
    pub query: Vec<(String, String)>, 
    pub body: Option<Body>, 
    pub retry: Option<RetryConfig>, 
    pub rate_limit: Option<RateLimitConfig> 
}

impl<Body: Clone> Endpoint<Body> {
    // Call to Constructor Helper 
    pub fn builder(path: impl Into<String>) -> EndpointBuilder<Body> {
        EndpointBuilder::new(path)
    }

    // Constructs Endpoint using Builder helper functions from config 
    pub fn from_config(config: &EndpointConfig) -> Self 
    where 
        Body: Default 
    {
        EndpointBuilder::new(config.path.clone())
            .query_params(config.query_defaults.clone())
            .retry_override(config.retry.clone())
            .rate_limit_override(config.rate_limit.clone())
            .finish() 
    }
}

/************ EndpointBuilder *****************************/ 
/* Opaque Struct which implements helper functions for 
 * building a single endpoint. 
 */
#[derive(Clone, Debug)]
pub struct EndpointBuilder<Body = serde_json::Value> {
    method: Method, 
    path: String, 
    query: Vec<(String, String)>, 
    body: Option<Body>, 
    retry: Option<RetryConfig>, 
    rate_limit: Option<RateLimitConfig> 
}


impl<Body> EndpointBuilder<Body> {
    // New/default 
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            method: Method::Get, 
            path: path.into(), 
            query: Vec::new(), 
            body: None, 
            retry: None, 
            rate_limit: None 
        }
    }

    pub fn method(mut self, method: Method) -> Self {
        self.method = method; 
        self 
    }

    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into(); 
        self 
    } 

    pub fn query_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((key.into(), value.into())); 
        self 
    }

    pub fn query_params<Item, Key, Value>(mut self, params: Item) -> Self 
    where 
        Item: IntoIterator<Item = (Key, Value)>,
        Key: Into<String>,
        Value: Into<String>
    {
        self.query.extend(params.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    pub fn body(mut self, body: Body) -> Self 
    where 
        Body: Serialize 
    {
        self.body = Some(body);
        self 
    }

    pub fn retry_override(mut self, retry: Option<RetryConfig>) -> Self {
        self.retry = retry; 
        self 
    }

    pub fn rate_limit_override(mut self, rate_limit: Option<RateLimitConfig>) -> Self {
        self.rate_limit = rate_limit; 
        self 
    }

    // Conversion from EndpointBuilder -> Endpoint
    pub fn finish(self) -> Endpoint<Body> {
        Endpoint {
            method: self.method, 
            path: self.path, 
            query: self.query, 
            body: self.body, 
            retry: self.retry, 
            rate_limit: self.rate_limit 
        }
    }
}

// Constructor from some Endpoint Seed into an Endpoint 
pub type EndpointConstructor<Seed, Body = serde_json::value> = 
    Arc<dyn Fn(&Seed) -> Endpoint<Body> + Send + Sync>; 

pub fn seed_endpoint_constructor<Seed, Body, F>(base: Endpoint<Body>, f: F)
    -> EndpointConstructor<Seed, Body> 
where 
    Body: Clone + Send + Sync + 'static, 
    F: Fn(&Seed, Endpoint<Body>) -> Endpoint<Body> + Send + Sync + 'static 
{
    Arc::new(move |seed| {
        let endp = base.clone(); 
        f(seed, endp)
    })
}
