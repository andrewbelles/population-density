-- Schema Bootstrap for climate.db 

PRAGMA foreign_keys = ON; 

-- Stations Table keyed on Station ID 
CREATE TABLE IF NOT EXISTS stations (
  station_id      TEXT PRIMARY KEY, 
  name            TEXT, 
  state           TEXT, 
  latitude        REAL, 
  longitude       REAL, 
  elevation_m     REAL, 
  active_start    DATE, 
  active_end      DATE, 
  last_ingested   DATETIME, 
  metadata        JSON
);

-- Daily observation data for a station keyed on id, date or id 
CREATE TABLE IF NOT EXISTS station_daily_obs (
  station_id      TEXT NOT NULL, 
  obs_date        DATE NOT NULL, 
  tmax_c          REAL, 
  tmin_C          REAL, 
  tavg_c          REAL, 
  prcp_mm         REAL, 
  snow_mm         REAL, 
  snwd_mm         REAL, 
  humidity_pct    REAL, 
  sunshine_pct    REAL, 
  source          TEXT, 
  updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP, 
  PRIMARY KEY (station_id, obs_date), 
  FOREIGN KEY (station_id) REFERENCES stations(station_id) ON DELETE CASCADE 
); 

CREATE INDEX IF NOT EXISTS idx_station_daily_obs_date 
  ON station_daily_obs (obs_date); 

-- Climate Normals per station  
CREATE TABLE IF NOT EXISTS station_normals (
  station_id    TEXT NOT NULL, 
  period_type   TEXT NOT NULL, 
  period_value  TEXT NOT NULL, 
  metric        TEXT NOT NULL, 
  value         REAL, 
  units         TEXT, 
  updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP, 
  PRIMARY KEY (station_id, period_type, period_value, metric), 
  FOREIGN KEY (station_id) REFERENCES stations(station_id) ON DELETE CASCADE
);

-- Division Metrics 
CREATE TABLE IF NOT EXISTS climate_division_metrics (
  division_id   TEXT NOT NULL, 
  year          INTEGER NOT NULL, 
  month         INTEGER NOT NULL, 
  metric        TEXT NOT NULL, 
  value         REAL, 
  units         TEXT, 
  updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP, 
  PRIMARY KEY (division_id, year, month, metric) 
); 

CREATE INDEX IF NOT EXISTS idx_division_metrics_div_year
  ON climate_division_metrics (division_id, year); 

-- ID of grids from nClimDiv 
CREATE TABLE IF NOT EXISTS grid_tiles (
  tile_id      TEXT PRIMARY KEY, 
  dataset      TEXT NOT NULL, 
  bbox         TEXT NOT NULL, 
  timestep     TEXT NOT NULL, 
  checksum     TEXT, 
  storage_path TEXT NOT NULL, 
  ingested_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tracks Jobs enqueued 
CREATE TABLE IF NOT EXISTS ingestion_runs (
  id           INTEGER PRIMARY KEY AUTOINCREMENT, 
  job_name     TEXT NOT NULL, 
  started_at   DATETIME NOT NULL, 
  completed_at DATETIME, 
  status       TEXT NOT NULL, 
  notes        TEXT, 
  rows_written INTEGER DEFAULT 0 
);


-- Quick views for monthly summaries and trends 

CREATE VIEW IF NOT EXISTS station_monthly_summaries AS 
SELECT 
  station_id, 
  CAST(strftime('%Y', obs_date) AS INTEGER) AS year, 
  CAST(strftime('%m', obs_date) AS INTEGER) AS month, 
  AVG(tmax_c) AS avg_tmax_c, 
  AVG(tmin_c) AS avg_tmin_c, 
  AVG(prcp_mm) AS avg_prcp_mm, 
  SUM(prcp_mm) AS total_prcp_mm, 
  SUM(snow_mm) AS total_snow_mm, 
  COUNT(*)     AS days_reported 
FROM station_daily_obs 
GROUP BY station_id, year, month; 

CREATE VIEW IF NOT EXISTS division_trends AS 
SELECT 
  division_id, 
  metric, 
  year, 
  AVG(value) AS avg_value, 
  SUM(value) AS total_value, 
  MIN(value) AS min_value, 
  MAX(value) AS max_value 
FROM climate_division_metrics 
GROUP BY division_id, metric, year; 
