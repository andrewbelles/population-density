///
/// crawler.rs  Andrew Belles  Dec 2nd, 2025 
///
/// Defines the Generic Crawler Parent Class for 
/// which all API Clients use to abstract calls to 
/// Databases and handling concurrency 
///

use std::marker::PhantomData; 

use async_trait::async_trait; 
use thiserror::Error; 

use crate::core::http::{HttpClient, HttpError};
use crate::core::storage::StorageError;
use crate::core::pagination::Pager; 

/************ PageBatch<T> ********************************/ 
/* A single batch resulting from a JSON Request.  
 *
 * Stores information for pagination::Pager to use to re-request 
 * at the same endpoint etc. 
 */
#[derive(Debug, Clone)]
pub struct PageBatch<T> {
    pub items: Vec<T>, 
    pub next_cursor: Option<String>, 
    pub has_more: Option<bool> 
}


/************ Errors **************************************/ 

/************ ParseError **********************************/ 
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("{0}")]
    Message(String)
}

/************ CrawlerError ********************************/ 
#[derive(Debug, Error)]
pub enum CrawlerError {
    #[error("http error: {0}")]
    Http(#[from] HttpError),
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),
    #[error("storage error: {0}")]
    Storage(#[from] StorageError)
}

/************ Generic Traits for Client Specific Handlers */ 

pub trait PageParser<R, Item> {
    fn parse(&self, raw: R) -> Result<PageBatch<Item>, ParseError>; 
}

#[async_trait]
pub trait Storage<Item> {
    async fn upsert_batch(&self, items: &[Item]) -> Result<(), StorageError>; 
}

/************ CrawlerStats ********************************/ 
/* Useful information about current runtime of Client */
#[derive(Debug, Default)]
pub struct CrawlerStats {
    pub pages: u32, 
    pub items: usize 
}

/************ Generic Interface ***************************/ 

/************ Crawler *************************************/ 
/* Generic Crawler Struct holding information on how to request data,
 * parse the resulting JSON, and store the parsed data in database  
 * 
 * Notes: 
 *   The http client has an 'a lifetime because Crawler does not
 *   explicitly own the client used to instantiate it, thus will be borrowed 
 *   by the Crawler without being owned by the Crawler 
 *
 *   The raw data fetched via the HttpClient is not explicitly used, however
 *   the compiler must not ignore its existence, allowing us to keep the Crawler 
 *   as generic as possible in regards to how we explicitly handle the raw data. 
 */
pub struct Crawler<'a, Raw, Item, P, S> 
where 
    P: PageParser<Raw, Item>, 
    S: Storage<Item>
{
    http: &'a HttpClient,    // HttpClient WILL be borrowed by the Crawler Client itself 
    path: String, 
    pager: Pager, 
    parser: P, 
    storage: S, 
    _raw: PhantomData<Raw>,  // Notifies that we do not explicitly use raw data but it is used 
    _item: PhantomData<Item> // Exact same idea 
}

impl<'a, Raw, Item, P, S> Crawler<'a, Raw, Item, P, S> 
where 
    Raw: Send + serde::de::DeserializeOwned, 
    Item: Send + Sync, 
    P: PageParser<Raw, Item> + Send + Sync, 
    S: Storage<Item> + Send + Sync 
{
    pub fn new(
        http: &'a HttpClient, path: String, pager: Pager, parser: P, storage: S) -> Self {
        Self {
            http, path, pager, parser, storage, _raw: PhantomData, _item: PhantomData 
        }
    }

    /******** Crawler::run ********************************/ 
    /* Runs the Crawler for a single cycle with support for pagination 
     *
     * We return: 
     *   CrawlerStats for # of items read and # of pages cursed over 
     */ 
    pub async fn run(&mut self) -> Result<CrawlerStats, CrawlerError> {
        let mut stats = CrawlerStats::default(); 

        loop {
            let params = match self.pager.next_params() {
                Some(p) => p, 
                None    => break,
            }; 

            let query: Vec<(&str, String)> =
                params.iter().map(|(k, v)| (k.as_str(), v.clone())).collect(); 

            let raw: Raw = self.http.get_json(&self.path, &query).await?; 
            let page = self.parser.parse(raw)?; 
            self.storage.upsert_batch(&page.items).await?; 

            let page_len = page.items.len(); 
            stats.pages += 1; 
            stats.items += page_len; 

            if !self.pager.advance_state(page.next_cursor, page.has_more, page_len) {
                break;
            }
        }

        Ok(stats)
    }
}
