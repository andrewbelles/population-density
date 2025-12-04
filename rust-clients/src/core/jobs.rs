///
/// jobs.rs  Andrew Belles  Dec 3rd, 2025 
///
///
///
///
///

use std::{marker::PhantomData, sync::Arc}; 
use async_trait::async_trait; 

use crate::core::{
    crawler::{Crawler, CrawlerError, CrawlerStats, PageParser},
    endpoint::EndpointConstructor, 
    pagination::Pager, 
    runner::Job, 
    storage::Storage, 
    http::HttpClient 
}; 


pub struct CrawlJob<Seed, Raw, Item, P, S, Body = serde_json::Value> 
where 
    P: PageParser<Raw, Item>, 
    S: Storage<Item> 
{
    pub seed: Seed, 
    pub endpoint_ctor: EndpointConstructor<Seed, Body>, 
    pub pager: Pager, 
    pub parser: P, 
    pub storage: S, 
    pub http: Arc<HttpClient>, 
    pub name: String,  
    _raw: PhantomData<Raw>,
    _item: PhantomData<Item>
}

#[async_trait]
impl<'a, Seed, Raw, Item, P, S, Body> Job for CrawlJob<Seed, Raw, Item, P, S, Body> 
where 
    Seed: Send + Sync + 'static, 
    Raw: Send + Sync + serde::de::DeserializeOwned + 'static,
    Item: Send + Sync + 'static, 
    P: PageParser<Raw, Item> + Send + Sync + 'static, 
    S: Storage<Item> + Send + Sync + Clone + 'static,
    Body: serde::Serialize + Clone + Send + Sync + 'static 
{
    type Output = CrawlerStats; 
    type Error  = CrawlerError; 

    async fn run(self: Box<Self>) -> Result<Self::Output, Self::Error> {
        let mut crawler = Crawler::new(
            self.http, 
            (self.endpoint_ctor)(&self.seed),
            self.pager, 
            self.parser, 
            self.storage
        ); 
        crawler.run().await 
    }

    fn name(&self) -> &str {
        &self.name
    }
}
