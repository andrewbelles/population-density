#!/usr/bin/env python3 
# 
# tensors.py  Andrew Belles  Jan 20th, 2026 
# 
# Image Tensor datasets representing Spatial Locations. Requires extra work to pack, tile, and 
# canvas counties together so that they are on uniform images for SpatialClassifier to use. 
# 

import fiona, csv, rasterio, rasterio.mask, argparse, sys, os, time

import numpy as np 

from pathlib import Path 

from typing import Tuple, Dict, Iterable, Protocol, Iterable  

from numpy.typing import NDArray

from abc import ABC, abstractmethod

from dataclasses import dataclass 

from torch.utils.data import Dataset, get_worker_info, Sampler 

from preprocessing.population_labels import build_label_map 

from scipy.ndimage import uniform_filter

from scipy.stats   import entropy 

from rasterio.warp import (
    transform_geom,
    reproject,
    Resampling
)

from rasterio.features import (
    rasterize 
)

from shapely.geometry import (
    shape,
    mapping 
)

from rasterio.errors import WindowError

from scipy.io import loadmat 

from utils.helpers import project_path, _mat_str_vector 

# --------------------------------------------------------- 
# Tiled MMap Loader 
# --------------------------------------------------------- 

class TileLoader(Dataset): 

    '''
    Memory-mapped tiled tensor dataset 

    Returns: (bag, label) by default 
    - bag: NDArray shape (max_bag_size, C, H, W), dtype float32  
    - label: int 
    - mask: bool array shape (max_bag_size,) 
    
    Optionally:
    - include fips in output 
    - include original (not subsampled) tile count 
    '''

    def __init__(
        self, 
        *,
        index_csv: str, 
        bin_path: str, 
        tile_shape: tuple[int, int, int], 
        max_bag_size: int, 
        dtype: np.dtype = np.float32,
        sample_frac: float | None = None, 
        random_state: int = 0, 
        shuffle_tiles: bool = True, 
        pad_value: float = 0.0, 
        return_fips: bool = False, 
        return_num_tiles: bool = False, 
        should_validate_index: bool = False 
    ): 
        self.index_csv             = index_csv
        self.bin_path              = bin_path
        self.tile_shape            = tuple(int(x) for x in tile_shape)
        self.tile_elems            = int(np.prod(self.tile_shape))
        self.max_bag_size          = int(max_bag_size)
        self.dtype                 = np.dtype(dtype)
        self.sample_frac           = sample_frac
        self.random_state          = random_state
        self.shuffle_tiles         = bool(shuffle_tiles)
        self.pad_value             = float(pad_value)
        self.return_fips           = bool(return_fips)
        self.return_num_tiles      = bool(return_num_tiles)
        self.should_validate_index = bool(should_validate_index)

        self._rng  = None 
        self.epoch = 0

        self.fips, self.labels, self.offset_elements, self.num_tiles = self.load_index() 

        self.mmap = None 

        if self.should_validate_index:
            self.validate_offsets() 

    def ensure_mmap(self): 
        if self.mmap is None: 
            self.mmap = np.memmap(self.bin_path, mode="r", dtype=self.dtype)

    def __getstate__(self): 
        state = self.__dict__.copy() 
        state["mmap"] = None 
        return state 

    def __len__(self) -> int: 
        return len(self.labels)

    def __getitem__(self, idx: int): 
        n_tiles = int(self.num_tiles[idx])
        label   = int(self.labels[idx])

        self.ensure_mmap()
        if self.mmap is None: 
            raise ValueError("mmap binary is not open")

        if n_tiles <= 0: 
            bag  = np.zeros((self.max_bag_size, *self.tile_shape), dtype=self.dtype)
            # mask = np.zeros((self.max_bag_size,), dtype=bool)
            return self.format_output(bag, label, idx, n_tiles)

        start = int(self.offset_elements[idx])
        end   = start + n_tiles * self.tile_elems 
        raw   = self.mmap[start:end]
        tiles = raw.reshape(n_tiles, *self.tile_shape)
        rng   = self.get_rng()
        keep  = n_tiles 

        if self.sample_frac is not None: 
            if not (0.0 < self.sample_frac <= 1.0): 
                raise ValueError("sample_frac must be in (0, 1]")
            keep = max(1, int(np.floor(n_tiles * self.sample_frac)))

        keep = min(keep, self.max_bag_size)
        if keep < n_tiles: 
            indices  = rng.choice(n_tiles, size=keep, replace=False)
            tiles    = tiles[indices] 

        if self.shuffle_tiles and keep > 1: 
            if tiles.flags.writeable:
                rng.shuffle(tiles)
            else: 
                perm  = rng.permutation(tiles.shape[0])
                tiles = tiles[perm] 

        '''
        if keep < self.max_bag_size:
            pad = np.full(
                (self.max_bag_size, *self.tile_shape), 
                self.pad_value, dtype=self.dtype
            )
            pad[:keep] = tiles 
            tiles      = pad 

        mask = np.zeros((self.max_bag_size,), dtype=bool)
        mask[:keep] = True 
        '''

        return self.format_output(tiles, label, idx, n_tiles)

    def close(self): 
        mm = getattr(self, "mmap", None)
        if mm is None: 
            return 
        try: 
            if hasattr(mm, "_mmap") and mm._mmap is not None: 
                mm._mmap.close() 
        finally: 
            self.mmap = None 

    def __del__(self): 
        try: 
            self.close()
        except Exception: 
            pass 

    def format_output(self, tiles, label, idx, n_tiles): 
        out = [tiles, label]
        if self.return_fips or self.return_num_tiles:
            if self.return_fips:
                out.append(self.fips[idx])
            if self.return_num_tiles:
                out.append(int(n_tiles))
        return tuple(out)

    def set_epoch(self, epoch: int): 
        self.epoch = int(epoch)
        self._rng  = None 

    def get_rng(self) -> np.random.Generator:
        if self._rng is not None: 
            return self._rng 

        worker_info = get_worker_info() 
        if worker_info is None: 
            base = self.random_state 
        else: 
            base = worker_info.seed ^ self.random_state

        base      = base ^ (self.epoch + 0x9E3779B9)
        self._rng = np.random.default_rng(base)
        return self._rng 

    def load_index(self): 

        fips    = []
        labels  = []
        offsets = []
        counts  = []

        with open(self.index_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f) 
            for row in reader: 
                f = (row.get("fips") or "").strip() 
                if f.isdigit():
                    f = f.zfill(5)

                label       = int(row["label"])
                byte_offset = int(row["byte_offset"])
                num_tiles   = int(row["num_tiles"])

                if byte_offset % self.dtype.itemsize != 0: 
                    raise ValueError("byte_offset is not aligned to dtype size")

                offsets.append(byte_offset // self.dtype.itemsize)
                counts.append(num_tiles)
                labels.append(label)
                fips.append(f)

        return (
            np.asarray(fips, dtype="U5"),
            np.asarray(labels, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(counts, dtype=np.int64)
        )

    def validate_offsets(self): 
        file_elems = os.path.getsize(self.bin_path) // self.dtype.itemsize 
        for i in range(len(self.labels)): 
            start   = int(self.offset_elements[i])
            n_tiles = int(self.num_tiles[i])
            end     = start + n_tiles * self.tile_elems 
            if end > file_elems:
                raise ValueError(f"index row {i} points past end of dataset.bin")

# ---------------------------------------------------------
# Geospatially Aware Sampler 
# ---------------------------------------------------------

class GeoBatchSampler(Sampler[list[int]]): 

    def __init__(
        self, 
        chunks: list[NDArray], 
        *,
        batch_size: int = 256, 
        chunks_per_batch: int = 16, 
        random_state: int = 0, 
        drop_last: bool = False 
    ): 
        if batch_size % chunks_per_batch != 0: 
            raise ValueError("batch_size must be divisible by chunks_per_batch")

        self.chunks = [np.asarray(c, dtype=np.int64) for c in chunks if len(c) > 0]
        if not self.chunks: 
            raise ValueError("no non-empty geo chunks")

        self.batch_size         = batch_size 
        self.chunks_per_batch   = chunks_per_batch
        self.samples_per_chunk  = self.batch_size // self.chunks_per_batch 
        self.random_state       = random_state 
        self.drop_last          = drop_last 
        self.epoch              = 0 
        self.n_items            = int(sum(len(c) for c in self.chunks))

    def set_epoch(self, epoch: int): 
        self.epoch = epoch 

    def __len__(self): 
        if self.drop_last: 
            return max(1, self.n_items // self.batch_size) 
        return max(1, int(np.ceil(self.n_items / self.batch_size)))

    def __iter__(self): 
        rng            = np.random.default_rng(self.random_state + 9973 * self.epoch)
        n_chunks       = len(self.chunks)
        replace_chunks = n_chunks < self.chunks_per_batch

        for _ in range(len(self)): 
            choice = rng.choice(n_chunks, size=self.chunks_per_batch, replace=replace_chunks)
            batch  = []
            for cid in choice: 
                chunk = self.chunks[int(cid)]
                srepl = len(chunk) < self.samples_per_chunk
                take  = rng.choice(chunk, size=self.samples_per_chunk, replace=srepl) 
                batch.extend(int(i) for i in take.tolist())
            yield batch 


class SpatialRowLoader(Dataset): 

    def __init__(
        self, 
        *,
        rows: NDArray, 
        labels: NDArray, 
        fips: NDArray, 
        coords: NDArray | None = None, 
        coords_path: str | None = None, 
        random_state: int = 0, 
        geo_chunk_size: int = 16, 
        geo_bins: int = 256, 
        return_index: bool = True, 
        return_fips: bool = False 
    ): 

        X = np.asarray(rows, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)

        if X.ndim != 2: 
            raise ValueError(f"rows must be 2d, got {X.shape}")
        if X.shape[0] != y.shape[0]: 
            raise ValueError(f"rows n={X.shape[0]} != labels n={y.shape[0]}")

        self.rows           = X 
        self.labels         = y 
        self.random_state   = random_state 
        self.geo_chunk_size = geo_chunk_size
        self.geo_bins       = geo_bins 
        self.return_index   = return_index
        self.return_fips    = return_fips

        f = np.asarray(fips).astype("U5").reshape(-1)
        if f.shape[0] != X.shape[0]: 
            raise ValueError("fips length mismatch") 
        self.fips = f 

        if coords is None: 
            self.coords = self.load_aligned_coords(coords_path)
        else: 
            c = np.asarray(coords, dtype=np.float32)
            if c.ndim != 2 or c.shape != (X.shape[0], 2): 
                raise ValueError(f"coords must be (N, 2), got {c.shape}")
            self.coords = c 

        self.geo_rank = self.build_geo_rank(self.coords, bins=self.geo_bins)

    def make_sampler(
        self, 
        *,
        indices: NDArray | None = None, 
        batch_size: int = 256, 
        chunks_per_batch: int = 16, 
        random_state: int = 0, 
        drop_last: bool = False 
    ): 
        chunks = self.build_geo_chunks(indices=indices, chunk_size=self.geo_chunk_size)
        return GeoBatchSampler(
            chunks, 
            batch_size=batch_size, 
            chunks_per_batch=chunks_per_batch,
            random_state=random_state, 
            drop_last=drop_last 
        )

    def __len__(self) -> int: 
        return self.rows.shape[0] 

    def __getitem__(self, idx: int): 
        out = [self.rows[idx], int(self.labels[idx])]
        if self.return_fips: 
            out.append(self.fips[idx])
        if self.return_index: 
            out.append(int(idx))
        return tuple(out)

    def build_geo_chunks(
        self, 
        indices: NDArray | None = None, 
        *, 
        chunk_size: int | None = None
    ): 
        if indices is None: 
            indices = np.arange(len(self), dtype=np.int64)
        else: 
            indices = np.asarray(indices, dtype=np.int64)

        csize   = int(self.geo_chunk_size if chunk_size is None else chunk_size)
        ordered = indices[np.argsort(self.geo_rank[indices], kind="stable")]

        return [ordered[i:i+csize] for i in range(0, ordered.size, csize) 
                if ordered[i:i+csize].size > 0]

    def load_aligned_coords(self, coords_path: str | None): 
        out = np.full((len(self.fips), 2), np.nan, dtype=np.float32)
        if coords_path is None: 
            coords_path = project_path("data", "datasets", "travel_proxy.mat")

        mat = loadmat(coords_path)
        if "fips_codes" not in mat or "coords" not in mat: 
            return out 

        fips   = _mat_str_vector(mat["fips_codes"]).astype("U5")
        coords = np.asarray(mat["coords"], dtype=np.float32)
        if coords.ndim == 2 and coords.shape == (2, fips.shape[0]): 
            coords = coords.T 
        if coords.ndim != 2 or coords.shape[1] != 2: 
            return out 

        idx_map = {f: i for i, f in enumerate(fips)}
        for i, f in enumerate(self.fips): 
            j = idx_map.get(f)
            if j is not None: 
                out[i] = coords[j]
        return out 

    @staticmethod 
    def build_geo_rank(coords: NDArray, bins: int = 256): 
        n    = coords.shape[0]
        rank = np.arange(n, dtype=np.int64)

        valid = np.isfinite(coords).all(axis=1)
        idx   = np.flatnonzero(valid)
        if idx.size == 0: 
            return rank 

        lat = coords[idx, 0]
        lon = coords[idx, 1]

        lat_span = max(float(lat.max() - lat.min()), 1e-6)
        lon_span = max(float(lon.max() - lon.min()), 1e-6)
        lat_bin  = np.clip(((lat - lat.min()) / lat_span * (bins - 1)).astype(np.int32), 
                           0, bins - 1)
        lon_norm = np.clip((lon - lon.min()) / lon_span, 0.0, 1.0)
        lon_key  = np.where((lat_bin % 2) == 0, lon_norm, 1.0 - lon_norm)
        
        order_local = np.lexsort((lon_key, lat_bin))
        order = idx[order_local]

        rank  = np.empty(n, dtype=np.int64)
        rank[order]  = np.arange(order.shape[0], dtype=np.int64)
        rank[~valid] = np.arange(order.shape[0], n, dtype=np.int64)
        return rank 

# ---------------------------------------------------------
# Binary Data Writer 
# ---------------------------------------------------------

@dataclass(frozen=True)
class CountyTileStream: 
    fips: str 
    label: int 
    tiles: Iterable[np.ndarray]

class TileStreamSource(Protocol): 
    '''
    Contract on how to produce per-county stream of tiles. 
    Writer does not care about how tiles are generated 
    '''

    def iter_bags(self) -> Iterable[CountyTileStream]: 
        ...

    @property
    def tile_shape(self) -> tuple[int, int, int]: 
        ...

    @property 
    def dtype(self) -> np.dtype: 
        ...


class BinaryTileWriter: 
    '''
    Writes monolithic dataset.bin + index.csv from any TileStreamSource 
    Streams tiles to avoid large in-memory bags 
    '''

    def __init__(
        self, 
        source: TileStreamSource,
        *,
        out_bin_path: str, 
        out_index_path: str, 
        empty_threshold: float = 1.0 # fraction of zeros in mask
    ): 
        self.source          = source 
        self.out_bin_path    = out_bin_path 
        self.out_index_path  = out_index_path 
        self.empty_threshold = empty_threshold

    def write(self): 

        tile_shape = self.source.tile_shape 
        dtype      = np.dtype(self.source.dtype)

        bytes_per_tile = dtype.itemsize 
        for d in tile_shape: 
            bytes_per_tile *= int(d)

        os.makedirs(os.path.dirname(self.out_bin_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.out_index_path), exist_ok=True)

        # For output print 
        t0        = time.perf_counter()
        last      = t0 
        bag_count = 0 

        with (open(self.out_bin_path, "wb") as bin_f, 
              open(self.out_index_path, "w", newline="", encoding="utf-8") as csv_f): 
            writer = csv.DictWriter(
                csv_f, 
                fieldnames=["fips", "label", "byte_offset", "num_tiles"]
            )
            writer.writeheader()

            byte_offset = 0 

            for bag in self.source.iter_bags():
                n_written = 0 
                
                for tile in bag.tiles: 
                    if tile.shape != tile_shape: 
                        raise ValueError(f"tile shape {tile.shape} != expected {tile_shape}")
                    if tile.dtype != dtype: 
                        tile = tile.astype(dtype, copy=False)

                    zero_frac = float((tile == 0).sum()) / float(tile.size)
                    if zero_frac >= self.empty_threshold:
                        continue 

                    bin_f.write(tile.tobytes(order="C"))
                    n_written += 1 

                if n_written == 0: 
                    continue 

                writer.writerow({
                    "fips": bag.fips, 
                    "label": int(bag.label),
                    "byte_offset": byte_offset,
                    "num_tiles": n_written
                })
                byte_offset += n_written * bytes_per_tile

                bag_count += 1 
                if bag_count % 200 == 0: 
                    now = time.perf_counter()
                    print(
                        f"[binary writer] {bag_count} bags | +{now - last:.2f}s | "
                        f"total {now - t0:.2f}s", file=sys.stderr, flush=True 
                    )
                    last = now 

# ---------------------------------------------------------
# Tensor Dataset Parent Class 
# ---------------------------------------------------------

class SpatialTensorDataset(ABC, TileStreamSource): 

    '''
    Parent Class for spatial tensor datasets (VIIRS, NLCD, TIGER).
    Handles label loading, county iteration, and persistence. 
    '''

    def __init__(
        self,
        counties_path: str | None = None, 
        label_year: int = 2010, 
        census_dir: str | Path = project_path("data", "census"), 
        max_counties: int | None = None, 
        tile_hw: tuple[int, int] = (256, 256)
    ): 
        if counties_path is None: 
            counties_path = project_path(
                "data", "geography", "county_shapefile", "tl_2020_us_county.shp"
            )

        self.counties_path = counties_path 
        self.tile_hw       = tuple(int(x) for x in tile_hw)
        self.max_counties  = max_counties if max_counties is None else int(max_counties)
        
        if label_year == 2013: 
            self.label_map, _ = build_label_map(2013, census_dir=census_dir)
        else: 
            _, edges          = build_label_map(2013, census_dir=census_dir)
            self.label_map, _ = build_label_map(label_year, train_edges=edges, census_dir=census_dir)
    
    def save(
        self,
        *,
        out_bin_path: str, 
        out_index_path: str, 
        empty_threshold: float = 1.0
    ):
        '''
        Wraps BinaryTileWriter
        '''
        writer = BinaryTileWriter(
            self,
            out_bin_path=out_bin_path,
            out_index_path=out_index_path,
            empty_threshold=empty_threshold
        )
        writer.write() 

    def iter_counties(
        self, 
        source: fiona.Collection,
        raster_crs 
    ) -> Iterable[Tuple[str, dict]]: 
        src_crs = source.crs 
        for feature in source: 
            props = feature.get("properties", {})
            fips  = str(props.get("GEOID", "")).strip() 
            if fips.isdigit(): 
                fips = fips.zfill(5)
            if not fips or fips not in self.label_map: 
                continue 

            geom = feature["geometry"]
            if src_crs and raster_crs and src_crs != raster_crs: 
                geom = transform_geom(src_crs, raster_crs, geom, precision=6)

            yield fips, geom

    def iter_tiles(self, data: NDArray) -> Iterable[NDArray]: 

        ht, wt = self.tile_hw 

        if data.ndim == 2: 
            h, w   = data.shape 
            for y0 in range(0, h, ht): 
                for x0 in range(0, w, wt): 
                    y1 = min(y0 + ht, h)
                    x1 = min(x0 + wt, w)

                    tile = data[y0:y1, x0:x1]

                    if tile.shape != (ht, wt):
                        padded = np.zeros((ht, wt), dtype=data.dtype)
                        padded[:tile.shape[0], :tile.shape[1]] = tile 
                        tile   = padded 

                    if not np.any(tile): 
                        continue 

                    yield tile

        elif data.ndim == 3: 
            c, h, w   = data.shape 
            for y0 in range(0, h, ht): 
                for x0 in range(0, w, wt): 
                    y1 = min(y0 + ht, h)
                    x1 = min(x0 + wt, w)

                    tile = data[:, y0:y1, x0:x1]

                    if tile.shape[-2:] != (ht, wt):
                        padded = np.zeros((c, ht, wt), dtype=data.dtype)
                        padded[:, :tile.shape[1], :tile.shape[2]] = tile 
                        tile   = padded 

                    if not np.any(tile): 
                        continue 

                    yield tile

        else: 
            return ValueError(f"expected 2d/3d data, got shape {data.shape}")

    @staticmethod 
    def tight_crop(data: NDArray, mask: NDArray) -> NDArray:
        ys, xs = np.nonzero(mask)
        if ys.size == 0: 
            if data.ndim == 3: 
                return np.zeros((data.shape[0], 0, 0), dtype=data.dtype)
            else: 
                return np.zeros((0, 0), dtype=np.float32)

        y0, y1 = ys.min(), ys.max() + 1 
        x0, x1 = xs.min(), xs.max() + 1 

        if data.ndim == 3: 
            return data[:, y0:y1, x0:x1]
        elif data.ndim == 2: 
            return data[y0:y1, x0:x1]
        else: 
            raise ValueError(f"expected 2d/3d shape, got shape {data.shape}")

    @abstractmethod
    def iter_bags(self):
        '''
        yields CountyTileStream objects. 
        Each stream yields tiles shaped (C, H, W), dtype == self.dtype 
        '''
        raise NotImplementedError

    @property 
    @abstractmethod 
    def tile_shape(self) -> tuple[int, int, int]: 
        '''
        Fixed tile shape (C, H, W) for specific dataset 
        '''
        raise NotImplementedError

    @property 
    def dtype(self) -> np.dtype: 
        return np.dtype(np.float32)

# ---------------------------------------------------------
# Viirs nighttime lights tensor dataset  
# ---------------------------------------------------------

class ViirsTensorDataset(SpatialTensorDataset):
    '''
    VIIRS tensor generator (single channel) 
    '''

    def __init__(
        self,
        *,
        viirs_path: str | None = None, 
        counties_path: str | None = None, 
        label_year: int = 2010, 
        census_dir: str | Path = project_path("data", "census"), 
        tile_hw: tuple[int, int] = (256, 256), 
        all_touched: bool = False, 
        log_scale: bool = True, 
        max_counties: int | None = None, 
        force_tight_crop: bool = False, 
    ): 
        super().__init__(
            counties_path=counties_path,
            label_year=label_year,
            census_dir=census_dir, 
            max_counties=max_counties,
            tile_hw=tile_hw
        )
        self.viirs_path        = viirs_path 
        self.force_tight_crop  = bool(force_tight_crop)
        self.all_touched       = bool(all_touched)
        self.log_scale         = bool(log_scale)

    @property
    def tile_shape(self) -> tuple[int, int, int]:
        ht, wt = self.tile_hw 
        return (2, ht, wt)

    @property 
    def dtype(self) -> np.dtype: 
        return np.dtype(np.float32)

    def iter_bags(self):
        if self.viirs_path is None: 
            raise ValueError("viirs_path is required")

        count = 0 
        with (rasterio.open(self.viirs_path) as src,
              fiona.open(self.counties_path) as shp): 
            
            for fips, geom in self.iter_counties(shp, src.crs): 
                try: 
                    out_image, _ = rasterio.mask.mask(
                        src, [geom], crop=True, filled=False, all_touched=self.all_touched
                    )
                # if geometry does not overlap w/ raster bounds, just move on
                except (ValueError, WindowError):
                    print("[viirs window error] skipping sample...", file=sys.stderr)
                    continue 

                arr = out_image[0]
                if arr.size == 0: 
                    continue 

                valid_mask = ~arr.mask if hasattr(arr, "mask") else np.isfinite(arr)
                if src.nodata is not None: 
                    valid_mask &= arr != src.nodata 

                if not np.any(valid_mask): 
                    continue 

                data = np.asarray(arr, dtype=np.float32)

                if self.log_scale: 
                    data = np.log1p(np.maximum(data, 0.0))
                    data = data / 9.0 

                # get local standard deviation across data 
                win_size = 3 
                mean     = uniform_filter(data, size=win_size, mode="reflect")
                sq_mean  = uniform_filter(data**2, size=win_size, mode="reflect")
                var      = sq_mean - mean**2 
                var      = np.maximum(var, 0.0)
                texture  = np.sqrt(var) # stdev 
                tile     = np.stack([data, texture], axis=0)

                tile[:, ~valid_mask] = 0.0 

                if self.force_tight_crop:
                    tile = self.tight_crop(tile, valid_mask)

                label      = self.label_map[fips]
                tiles_iter = (t for t in self.iter_tiles(tile)) 

                yield CountyTileStream(
                    fips=fips,
                    label=label,
                    tiles=tiles_iter, 
                )

                count += 1 
                if self.max_counties is not None and count >= self.max_counties: 
                    break 

# ---------------------------------------------------------
# United States Postal Service vacant and active addresses.  
# ---------------------------------------------------------

class UspsTensorDataset(SpatialTensorDataset): 
    
    '''
    Image dataset per county of active and vacant residential and business addresses, meant 
    to address gap in model architecture for assisting in middle class 
    identification (Suburban ranges for NCHS). 
    '''

    def __init__(
        self,
        *,
        usps_vector_path: str | None = None, 
        reference_path: str | None = None,   # VIIRS raster as grid reference 
        nlcd_path: str | None = None,        # suppression of water pixels in mask 
        counties_path: str | None = None, 
        labels_path: str | None = None, 
        tile_hw: tuple[int, int] = (256, 256), 
        usps_layer: str | None = None, 
        water_codes: tuple[int, ...] = (11, 12), 
        developed_codes: tuple[int, ...] = (21, 22, 23, 24), 
        all_touched: bool = False, 
        max_counties: int | None = None, 
        force_tight_crop: bool = False 
    ): 
        super().__init__(
            counties_path=counties_path,
            labels_path=labels_path,
            max_counties=max_counties,
            tile_hw=tile_hw 
        )

        self.usps_vector_path = usps_vector_path 
        self.reference_path   = reference_path 
        self.nlcd_path        = nlcd_path 
        self.usps_layer       = usps_layer 
        self.all_touched      = bool(all_touched) 
        self.water_codes      = tuple(int(x) for x in water_codes)
        self.developed_codes  = tuple(int(x) for x in developed_codes)
        self.force_tight_crop = bool(force_tight_crop)

    @property
    def tile_shape(self) -> tuple[int, int, int]: 
        ht, wt = self.tile_hw 
        return (4, ht, wt)

    @property 
    def dtype(self) -> np.dtype: 
        return np.dtype(np.float32)

    def iter_bags(self):
        if self.reference_path is None: 
            raise ValueError("reference path is required")
        if self.usps_vector_path is None: 
            raise ValueError("usps_vector_path is required")
        if self.nlcd_path is None: 
            raise ValueError("nlcd_path is required")
        if self.counties_path is None: 
            raise ValueError("counties_path is required")
        if self.labels_path is None: 
            raise ValueError("labels_path is required")

        count = 0 
        skipped = 0
        with (rasterio.open(self.reference_path) as ref, 
              rasterio.open(self.nlcd_path) as nlcd, 
              fiona.open(self.usps_vector_path, layer=self.usps_layer) as tracts,
              fiona.open(self.counties_path) as shp): 

            tracts_crs = tracts.crs 

            for fips, geom in self.iter_counties(shp, ref.crs): 
                try: 
                    out_image, out_transform = rasterio.mask.mask(
                        ref, [geom], crop=True, filled=False, all_touched=self.all_touched,
                    )
                except (ValueError, WindowError): 
                    print("[usps window error] skipping sample...", file=sys.stderr)
                    continue 

                arr = out_image[0]
                if arr.size == 0: 
                    continue 

                valid_mask = ~arr.mask if hasattr(arr, "mask") else np.isfinite(arr)
                if ref.nodata is not None: 
                    valid_mask &= arr != ref.nodata 
                if not np.any(valid_mask): 
                    continue 

                H, W     = arr.shape 
                channels = np.zeros((4, H, W), dtype=np.float32) 

                nlcd_dst = np.zeros((H, W), dtype=nlcd.dtypes[0])

                reproject(
                    source=rasterio.band(nlcd, 1), 
                    destination=nlcd_dst,
                    src_transform=nlcd.transform,
                    src_crs=nlcd.crs,
                    dst_transform=out_transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.nearest,
                    dst_nodata=nlcd.nodata 
                )

                nlcd_valid     = (nlcd_dst != nlcd.nodata if nlcd.nodata is not None 
                                  else np.ones_like(nlcd_dst, bool))
                water_mask     = np.isin(nlcd_dst, self.water_codes) & nlcd_valid 
                land_mask      = (~water_mask) & nlcd_valid 

                county_shape   = shape(geom)
                bbox           = county_shape.bounds 
                
                for feat in tracts.filter(bbox=bbox): 

                    props        = feat.get("properties") or {}
                    bus_vac_rate = self._as_float(props, "bus_vac_rate")
                    comm_ratio   = self._as_float(props, "comm_ratio")
                    vac_rate     = self._as_float(props, "vac_rate")
                    flux_rate    = self._as_float(props, "flux_rate")

                    if (bus_vac_rate is None or flux_rate is None or 
                        comm_ratio is None or vac_rate is None): 
                        skipped += 1
                        continue 

                    tgeom = feat.get("geometry")
                    if tgeom is None: 
                        continue 

                    if tracts_crs and ref.crs and tracts_crs != ref.crs: 
                        tgeom = transform_geom(tracts_crs, ref.crs, tgeom, precision=6)

                    tract_shape = shape(tgeom)
                    if not tract_shape.intersects(county_shape): 
                        continue 

                    inter = tract_shape.intersection(county_shape) 
                    if inter.is_empty: 
                        continue 

                    tract_mask = rasterize(
                        [(mapping(inter), 1)],
                        out_shape=(H, W), 
                        transform=out_transform,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    ).astype(bool)

                    if not tract_mask.any(): 
                        continue 

                    target = tract_mask & land_mask 

                    if target.any(): 
                        tract_land_cover = nlcd_dst[target]

                        if tract_land_cover.size > 0: 
                            _, counts = np.unique(tract_land_cover, return_counts=True)
                            texture   = entropy(counts)
                        else: 
                            texture   = 0.0 

                        channels[0, target] = comm_ratio 
                        channels[1, target] = vac_rate 
                        channels[2, target] = texture  
                        channels[3, target] = flux_rate 

                mask = (valid_mask & nlcd_valid).astype(np.uint8)
                if not np.any(mask):
                    continue 

                if self.force_tight_crop: 
                    channels = self.tight_crop(channels, mask)

                tiles_iter = (tile for tile in self.iter_tiles(channels))
                label      = self.label_map[fips]

                yield CountyTileStream(
                    fips=fips,
                    label=label,
                    tiles=tiles_iter
                ) 

                count += 1 
                if self.max_counties is not None and count >= self.max_counties:
                    break 

    @staticmethod 
    def _as_float(props, key): 
        v = props.get(key)
        if v is None: 
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None 
        if not np.isfinite(v): 
            return None 
        return v

# ---------------------------------------------------------
# Main entry point  
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--viirs-path", default=project_path(
        "data", "viirs", "viirs_2013_median_masked.tif"))
    parser.add_argument("--nlcd-path", default=project_path(
        "data", "nlcd", "Annual_NLCD_LndCov_2023_CU_C1V1.tif"))
    parser.add_argument("--usps-gpkg", default=project_path(
        "data", "usps", "usps_master_tracts.gpkg"))

    parser.add_argument("--counties-path", default=project_path(
        "data", "geography", "county_shapefile", "tl_2020_us_county.shp"))
    parser.add_argument("--labels-path", default=project_path(
        "data", "nchs", "nchs_classification_2013.csv"))

    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--census-dir", default=project_path("data", "census"))

    parser.add_argument("--out-root", default=project_path("data", "tensors"))
    parser.add_argument("--viirs-out", default=None)
    parser.add_argument("--usps-out", default=None)

    parser.add_argument("--no-log-scale", action="store_true")
    
    # hooks to run
    parser.add_argument("--viirs", action="store_true")
    parser.add_argument("--usps", action="store_true")
    args = parser.parse_args()

    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    viirs_out = args.viirs_out or str(out_root / "viirs")
    usps_out  = args.usps_out  or str(out_root / "usps")

    if args.viirs:
        bin_out   = Path(viirs_out) / "dataset.bin" 
        index_out = Path(viirs_out) / "index.csv"

        viirs = ViirsTensorDataset(
            viirs_path=args.viirs_path,
            counties_path=args.counties_path,
            label_year=args.year,
            census_dir=args.census_dir, 
            log_scale=not args.no_log_scale,
        )
        viirs.save(out_bin_path=str(bin_out), out_index_path=str(index_out))

    if args.usps:
        bin_out   = Path(usps_out) / "dataset.bin" 
        index_out = Path(usps_out) / "index.csv"

        usps = UspsTensorDataset(
            usps_vector_path=args.usps_gpkg, 
            reference_path=args.viirs_path,
            nlcd_path=args.nlcd_path,
            counties_path=args.counties_path,
            labels_path=args.labels_path,
        )
        usps.save(out_bin_path=str(bin_out), out_index_path=str(index_out))

if __name__ == "__main__": 
    main() 
