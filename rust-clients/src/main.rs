///
/// main.rs  Andrew Belles  Dec 4th, 2025 
///
/// Main execution of API Clients through Parent Daemon Service 
///
///
///

mod core; 
mod parent; 
mod clients; 

use std::sync::Arc;

use clients::station::{StationParser, station_storage};

use core::{
    config::PaginationConfig, 
    endpoint::{Endpoint, seed_endpoint_constructor}, 
    http::HttpClient, 
    jobs::CrawlJob, 
    pagination::Pager, 
    runner::{Job, Runner}
};

use parent::{
    daemon::{spawn_daemon, ClientSpec},
    init::load_client_from_yaml 
}; 

#[derive(Clone, Copy)]
struct NoopJob(&'static str);

#[async_trait::async_trait]
impl Job for NoopJob {
    type Output = (); 
    type Error  = (); 

    async fn run(self: Box<Self>) -> Result<Self::Output, Self::Error> {
        Ok(())
    }

    fn name(&self) -> &str {
        self.0 
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config  = load_client_from_yaml("config.yaml", "station")?; 
    
    println!("{:?}", config);

    let http    = Arc::new(HttpClient::new(config.http.clone())?); 
    let storage = station_storage(&config.storage).await?; 
    let parser  = StationParser; 

    let endp_cfg = config.endpoint_overrides 
        .get("stations")
        .expect("station endpoint config missing"); 
    let endpoint: Endpoint = Endpoint::from_config(endp_cfg); 
    let pager = Pager::new(endp_cfg.pagination.clone().unwrap_or_else(PaginationConfig::default)); 

    let ctor = seed_endpoint_constructor(endpoint, |_seed: &(), endp| endp); 

    let crawl_job = CrawlJob {
        seed: (), 
        endpoint_ctor: ctor, 
        pager, 
        parser, 
        storage, 
        http: http.clone(), 
        name: "stations".to_string(), 
        _raw: std::marker::PhantomData, 
        _item: std::marker::PhantomData 
    };
    
    println!("{:?}", crawl_job);

    let mut runner = Runner::<NoopJob>::new(config.runner.clone()); 
    runner.start(); 

    let stations_rate = endp_cfg.rate_limit 
        .as_ref() 
        .expect("rate_limit required for stations");

    let client_spec = ClientSpec {
        id: "station".to_string(),
        deps: vec![], 
        max_tokens: stations_rate.burst, 
        refill_per_minute: stations_rate.requests_per_minute
    }; 

    println!("{:?}", client_spec);

    let (_transaction, daemon_handle) = spawn_daemon(runner, &[client_spec], &[]); 

    tokio::time::sleep(std::time::Duration::from_millis(100)).await; 
    drop(_transaction); 

    let _ = daemon_handle.await; 
    Ok(())
}
