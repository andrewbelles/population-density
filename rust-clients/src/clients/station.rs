///
/// station.rs  Andrew Belles  Dec 4th, 2025 
///
/// Implementation of NWS Station Client, collecting  
/// information for stations such as geographic location,
/// station id, what datasets they have avaliable 
///

use std::sync::Arc;

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc}; 
use serde::{Deserialize, Serialize}; 
use serde_json::Value; 
use sqlx::{Executor, types::Json, QueryBuilder, Sqlite, Transaction}; 
use futures::future::BoxFuture; 

use crate::core::{
    config::StorageConfig, 
    crawler::{PageBatch, PageParser, ParseError}, 
    storage::{SqliteStorage, StorageError}
};

/*********** Static, Global Constants *********************/ 

pub const STATION_SOURCE: &str = "noaa"; 
const FEET_TO_METERS: f64 = 0.3048; 

/************ Queries on SqliteDB *************************/ 

pub const INSERT: &str = 
r#"
INSERT INTO stations (
    station_id, source, name, state, country, latitude, longitude, 
    elevation_m, datacoverage, mindate, maxdate, network, station_type,
    last_seen, last_ingested, metadata
)
VALUES
"#;

pub const CONFLICT: &str = 
r#"
ON CONFLICT(station_id) DO UPDATE SET 
    source = excluded.source, 
    name          = excluded.name, 
    state         = excluded.state, 
    country       = excluded.country,
    latitude      = excluded.latitude,
    longitude     = excluded.longitude,
    elevation_m   = excluded.elevation_m,
    datacoverage  = excluded.datacoverage,
    mindate       = excluded.mindate,
    maxdate       = excluded.maxdate,
    network       = excluded.network,
    station_type  = excluded.station_type,
    last_seen     = excluded.last_seen,
    last_ingested = excluded.last_ingested,
    metadata      = excluded.metadata,
    updated_at    = CURRENT_TIMESTAMP
"#;

/************ Station *************************************/ 
/* Single Station Entry into stations.sqlite */ 
#[derive(Debug, Clone)]
pub struct Station {
    pub station_id: String, 
    pub source: String, 
    pub name: Option<String>, 
    pub state: Option<String>, 
    pub country: Option<String>, 
    pub latitude: f64, 
    pub longitude: f64, 
    pub elevation_m: Option<f64>, 
    pub datacoverage: Option<f64>,
    pub mindate: Option<NaiveDate>, 
    pub maxdate: Option<NaiveDate>,
    pub network: Option<String>, 
    pub station_type: Option<String>, 
    pub last_seen: Option<NaiveDateTime>, 
    pub last_ingested: DateTime<Utc>, 
    pub metadata: Value 
}

/************ StationResponse *****************************/ 
/* Serialized response from HttpClient::get_json */ 
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StationResponse {
    #[serde(default)]
    pub results: Vec<StationPayload>, 
    #[serde(default)]
    pub metadata: Option<StationMeta> 
}

/************ StationMeta *********************************/ 
/* Wrapper over JSON response */ 
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StationMeta {
    #[serde(default)]
    pub resultset: Option<Resultset> 
}

/************ Resultset ***********************************/ 
/* Pagination helper struct for determining if response is 
 * a chunk of a page.  
 */ 
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Resultset {
    pub offset: Option<u32>, 
    pub count: Option<u32>, 
    pub limit: Option<u32> 
}

/************ StationPayload ******************************/ 
/* Raw deserialized data from JSON response */ 
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StationPayload {
    pub id: String, 
    pub name: Option<String>, 
    pub state: Option<String>, 
    pub country: Option<String>, 
    pub latitude: f64, 
    pub longitude: f64, 
    pub elevation: Option<f64>, 
    pub elevation_unit: Option<String>, 
    pub datacoverage: Option<f64>,
    pub mindate: Option<NaiveDate>, 
    pub maxdate: Option<NaiveDate>,
    #[serde(default)]
    pub station: Option<String> 
}

/************ StationParser *******************************/ 
/* Wrapper for Implementation of PageParser::parse */ 
#[derive(Clone, Default)]
pub struct StationParser; 

impl StationPayload {
    fn into_station(self, now: DateTime<Utc>) -> Result<Station, ParseError> {
        let metadata = serde_json::to_value(&self)
            .map_err(|e| ParseError::Message(format!("serialize station metadata: {e}")))?;

        let StationPayload {
            id, 
            name, 
            state, 
            country, 
            latitude, 
            longitude, 
            elevation, 
            elevation_unit, 
            datacoverage, 
            mindate, 
            maxdate, 
            station
        } = self; 

        let elevation_m = match (elevation, elevation_unit.as_deref()) {
            (Some(val), Some(unit)) if unit.eq_ignore_ascii_case("feet") => Some(val * FEET_TO_METERS),
            (Some(val), _) => Some(val),
            _ => None 
        };

        let last_seen = maxdate.and_then(|d| d.and_hms_opt(0, 0, 0));
        let network   = id.split(':').next().map(|s| s.to_string());
        let station_type = station;
        let datacoverage = datacoverage.map(|v| v.clamp(0.0, 1.0)); 

        Ok(Station {
            station_id: id, 
            source: STATION_SOURCE.to_string(), 
            name, 
            state, 
            country, 
            latitude, 
            longitude,
            elevation_m, 
            datacoverage, 
            mindate,
            maxdate,
            network,
            station_type, 
            last_seen,
            last_ingested: now, 
            metadata 
        })
    }
}

impl PageParser<StationResponse, Station> for StationParser {
    fn parse(&self, raw: StationResponse) -> Result<PageBatch<Station>, ParseError> {
        let now = Utc::now(); 
        let mut items = Vec::with_capacity(raw.results.len()); 

        for payload in raw.results {
            items.push(payload.into_station(now)?);
        }

        let has_more = raw.metadata.as_ref()
            .and_then(|m| m.resultset.as_ref())
            .and_then(|rs| {
                let count  = rs.count?; 
                let offset = rs.offset.unwrap_or(0); 
                let limit  = rs.limit.unwrap_or(items.len() as u32); 
                Some(offset + limit < count)
            });

        Ok(PageBatch {
            items, 
            next_cursor: None, 
            has_more
        }) 
    }
}

// Satisfying the retched higher ranking trait bounds that I pigeonhole'd myself into 
// like a fucking idiot. 
async fn upsert_station_batch(transaction: & mut Transaction<'_, Sqlite>, stations: & [Station]) 
    -> Result<(), StorageError> {
    // Early return for empty response 
    if stations.is_empty() {
        return Ok(());
    }

    // Build query in parellel over all Stations in batch 

    let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(INSERT); 

    qb.push_values(stations, |mut b, station| {
        b.push_bind(&station.station_id)
         .push_bind(&station.source)
         .push_bind(&station.name)
         .push_bind(&station.state)
         .push_bind(&station.country)
         .push_bind(station.latitude)
         .push_bind(station.longitude)
         .push_bind(station.elevation_m)
         .push_bind(station.datacoverage)
         .push_bind(station.mindate)
         .push_bind(station.maxdate)
         .push_bind(&station.network)
         .push_bind(&station.station_type)
         .push_bind(station.last_seen)
         .push_bind(station.last_ingested)
         .push_bind(Json(station.metadata.clone()));
    });
    
    qb.push(CONFLICT); 

    transaction.execute(qb.build()).await?; 
    Ok(())
}

// Same "issue" (annoyance) as upsert 
pub async fn station_storage(config: &StorageConfig) 
    -> Result<Arc<SqliteStorage<Station>>, StorageError> {
    let storage = SqliteStorage::new(config, |transaction, batch| {
        Box::pin(upsert_station_batch(transaction, batch))
    }).await?;  
    Ok(Arc::new(storage))
}
