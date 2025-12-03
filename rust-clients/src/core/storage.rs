///
/// storage.rs  Andrew Belles  Dec 2nd, 2025 
/// 
///
///
///
/// 

use std::sync::Arc; 

use async_trait::async_trait;

use futures::future::BoxFuture; 
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode},
    Pool, Sqlite, Transaction
};

use thiserror::Error; 

use crate::core::config::StorageConfig;


/************ StorageError ********************************/ 
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("sqlx error: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error("migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError), 
    #[error("storage init error: {0}")]
    Init(String),
}

/************ WriteBatch<Item> ****************************/ 
/* Type alias for upsert signature on generic Item */ 
type WriteBatchFn<Item> = dyn for<'a> Fn(&mut Transaction<'a, Sqlite>, &'a [Item]) 
    -> BoxFuture<'a, Result<(), StorageError>> + Send + Sync; 

pub struct SqliteStorage<Item> {
    pool: Pool<Sqlite>, 
    write_batch: Arc<WriteBatchFn<Item>>
}

impl<Item> SqliteStorage<Item> {
    pub async fn new<F>(config: &StorageConfig, write_batch: F) -> Result<Self, StorageError>
    where 
        F: for<'a> Fn(&mut Transaction<'a, Sqlite>, &'a [Item]) -> BoxFuture<'a, Result<(), StorageError>>  
            + Send + Sync + 'static 
    {
        let mut opts = SqliteConnectOptions::new() 
            .filename(&config.db_path)
            .create_if_missing(true);
        
        if let Some(ms) = config.busy_timeout_ms {
            opts = opts.busy_timeout(std::time::Duration::from_millis(ms));
        }

        if let Some(journal) = &config.journal_mode {
            let mode = match journal.as_str() {
                "WAL"      => SqliteJournalMode::Wal, 
                "MEMORY"   => SqliteJournalMode::Memory,
                "OFF"      => SqliteJournalMode::Off, 
                "DELETE"   => SqliteJournalMode::Delete, 
                "TRUNCATE" => SqliteJournalMode::Truncate,
                _ => return Err(StorageError::Init(format!("unknown journal mode: {}", journal)))
            };
            opts = opts.journal_mode(mode); 
        }
        
        // Initalize pool with options 
        let mut pool_opts = sqlx::sqlite::SqlitePoolOptions::new(); 
        if let Some(max) = config.max_connections {
            pool_opts = pool_opts.max_connections(max); 
        }
        let pool = pool_opts.connect_with(opts).await?; 

        Ok(Self { pool, write_batch: Arc::new(write_batch) })
    }

    pub fn pool(&self) -> &Pool<Sqlite> {
        &self.pool
    }

    pub async fn migrate(&self, migrator: &sqlx::migrate::Migrator) -> Result<(), StorageError> {
        migrator.run(&self.pool).await?; 
        Ok(())
    }
    pub async fn health_check(&self) -> Result<(), StorageError> {
        sqlx::query("select 1").execute(&self.pool).await?; 
        Ok(())
    }
}

#[async_trait]
impl<T, Item> crate::core::crawler::Storage<Item> for Arc<T> 
where 
    T: crate::core::crawler::Storage<Item> + Send + Sync + ?Sized, 
    Item: Send + Sync 
{
    async fn upsert_batch(&self, items: &[Item]) -> Result<(), StorageError> {
        (**self).upsert_batch(items).await
    }
}

#[async_trait]
impl<Item: Sync + Send> crate::core::crawler::Storage<Item> for SqliteStorage<Item> {
    async fn upsert_batch(&self, items: &[Item]) -> Result<(), StorageError> {
        let mut transaction = self.pool.begin().await.map_err(StorageError::from)?; 
        (self.write_batch)(&mut transaction, items).await?;
        transaction.commit().await?; 
        Ok(())
    }
}
