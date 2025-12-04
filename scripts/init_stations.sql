-----------------------------------------------------------
-- init_stations.sql  Andrew Belles  Dec 3rd, 2025 
-- 
-- Initializes the data/stations.sqlite database 
-- Should match upsert_batch for climate client in rust-clients 
-- 
-----------------------------------------------------------

-- Initialize the station catalog database used for station lookups
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

BEGIN;

-- Track ingestion runs for auditing
CREATE TABLE IF NOT EXISTS ingestion_runs (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  job_name     TEXT NOT NULL,
  started_at   DATETIME NOT NULL,
  completed_at DATETIME,
  status       TEXT NOT NULL,
  notes        TEXT,
  rows_written INTEGER DEFAULT 0
);

-- Core station metadata, intended to be shared with downstream databases
CREATE TABLE IF NOT EXISTS stations (
  station_id    TEXT PRIMARY KEY,
  source        TEXT NOT NULL DEFAULT 'noaa',
  name          TEXT,
  state         TEXT,
  country       TEXT,
  latitude      REAL NOT NULL CHECK (latitude BETWEEN -90.0 AND 90.0),
  longitude     REAL NOT NULL CHECK (longitude BETWEEN -180.0 AND 180.0),
  elevation_m   REAL,
  datacoverage  REAL CHECK (datacoverage BETWEEN 0.0 AND 1.0),
  mindate       DATE,
  maxdate       DATE,
  network       TEXT, -- program/network label from the upstream API
  station_type  TEXT, -- additional classifier
  last_seen     DATETIME, -- last time the source reported this station
  last_ingested DATETIME,
  metadata      JSON, -- raw upstream payload
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stations_source ON stations(source);
CREATE INDEX IF NOT EXISTS idx_stations_state ON stations(state);
CREATE INDEX IF NOT EXISTS idx_stations_country ON stations(country);
CREATE INDEX IF NOT EXISTS idx_stations_geo ON stations(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_stations_last_seen ON stations(last_seen);

-- Canonical export shape for downstream databases
CREATE VIEW IF NOT EXISTS station_export AS
SELECT
  station_id,
  source,
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
  last_ingested
FROM stations;

COMMIT;
