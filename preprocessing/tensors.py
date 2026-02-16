#!/usr/bin/env python3 
# 
# tensors.py  Andrew Belles  Jan 20th, 2026 
# 
# Image Tensor datasets representing Spatial Locations. Requires extra work to pack, tile, and 
# canvas counties together so that they are on uniform images for SpatialClassifier to use. 
# 

import fiona, csv, rasterio, rasterio.mask, argparse, sys, os, time

import numpy                as np 

from pathlib                import (
    Path 
)

from typing                 import (
    Tuple, 
    Iterable, 
    Protocol, 
    Iterable  
)

from numpy.typing           import (
    NDArray
)

from abc                    import (
    ABC, 
    abstractmethod
)

from dataclasses            import (
    dataclass 
)

from torch.utils.data       import (
    Dataset, 
    get_worker_info, 
    Sampler 
)

from preprocessing.labels   import (
    build_label_map 
)

from rasterio.warp          import (
    transform_geom,
)

from rasterio.errors        import (
    WindowError
)

from scipy.io               import (
    loadmat 
)

from utils.helpers          import (
    project_path,
    _mat_str_vector 
)

# ---------------------------------------------------------
# Patch Statistics 
# ---------------------------------------------------------

def tile_patch_stats(tiles: np.ndarray, patch_size: int = 32) -> np.ndarray:
    c, h, w = tiles.shape 
    p = patch_size 

    gh, gw = h // p, w // p 
    patches = tiles.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, c, p * p)
    return np.quantile(patches, 0.95, axis=-1).astype(np.float32)

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
        stats_bin_path: str, 
        tile_shape: tuple[int, int, int], 
        patch_size: int = 32, 
        dtype=np.float32,
        stats_dtype=np.float32,
        random_state: int = 0, 
        return_fips: bool = False, 
        return_num_tiles: bool = False, 
        should_validate_index: bool = False 
    ): 
        self.index_csv             = index_csv
        self.bin_path              = bin_path
        self.stats_bin_path        = stats_bin_path

        self.tile_shape            = tuple(int(x) for x in tile_shape)
        self.tile_elems            = int(np.prod(self.tile_shape))
        self.patch_size            = patch_size 

        self.dtype                 = np.dtype(dtype)
        self.stats_dtype           = np.dtype(stats_dtype)

        self.random_state          = random_state
        self.return_fips           = bool(return_fips)
        self.return_num_tiles      = bool(return_num_tiles)
        self.should_validate_index = bool(should_validate_index)

        self.image_bytes_per_tile  = self.tile_elems * self.dtype.itemsize 
        self.num_patches_per_tile  = (self.tile_shape[1] // self.patch_size)**2 # assume square 
        self.stats_dim             = self.tile_shape[0] 
        self.stats_elems_per_tile  = self.num_patches_per_tile * self.stats_dim 

        self._rng  = None 
        self.epoch = 0

        self.fips, self.labels, self.offset_bytes, self.num_tiles = self.load_index() 
        self.offset_elements = self.offset_bytes // self.dtype.itemsize 

        self.stats_mmap = None 
        self.mmap       = None 

        if self.should_validate_index:
            self.validate_offsets() 

    def ensure_mmaps(self): 
        if self.mmap is None: 
            self.mmap = np.memmap(self.bin_path, mode="r", dtype=self.dtype)
        if self.stats_mmap is None: 
            self.stats_mmap = np.memmap(self.stats_bin_path, mode="r", dtype=self.stats_dtype)

    def __getstate__(self): 
        state = self.__dict__.copy() 
        state["mmap"]       = None 
        state["stats_mmap"] = None 
        return state 

    def __len__(self) -> int: 
        return len(self.labels)

    def __getitem__(self, idx: int): 
        n_tiles = int(self.num_tiles[idx])
        label   = float(self.labels[idx])

        self.ensure_mmaps()
        if self.mmap is None or self.stats_mmap is None: 
            raise ValueError("mmap binaries are not open")

        if n_tiles <= 0: 
            tiles = np.zeros((0, *self.tile_shape), dtype=self.dtype)
            stats = np.zeros((0, self.num_patches_per_tile, self.stats_dim), 
                             dtype=self.stats_dtype)
            return self.format_output(tiles, stats, label, idx, n_tiles)

        byte_offset = int(self.offset_bytes[idx])

        start_elem  = byte_offset // self.dtype.itemsize 
        end_elem    = start_elem + n_tiles * self.tile_elems 
        tiles       = self.mmap[start_elem:end_elem].reshape(n_tiles, *self.tile_shape)

        tile_start  = byte_offset // self.image_bytes_per_tile 
        stats_start = tile_start * self.stats_elems_per_tile 
        stats_end   = stats_start + n_tiles * self.stats_elems_per_tile 
        stats       = (self.stats_mmap[stats_start:stats_end]
                       .reshape(n_tiles, self.num_patches_per_tile, self.stats_dim))

        return self.format_output(tiles, stats, label, idx, n_tiles)

    def close(self): 
        for name in ("mmap", "stats_mmap"): 
            mm = getattr(self, name, None)
            if mm is None: 
                return 
            try: 
                if hasattr(mm, name) and mm._mmap is not None: 
                    mm._mmap.close() 
            finally: 
                setattr(self, name, None) 

    def __del__(self): 
        try: 
            self.close()
        except Exception: 
            pass 

    def format_output(self, tiles, stats, label, idx, n_tiles): 
        out = [tiles, stats, label]
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

        fips         = []
        labels       = []
        byte_offsets = []
        counts       = []

        with open(self.index_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f) 
            for row in reader: 
                f = (row.get("fips") or "").strip() 
                if f.isdigit():
                    f = f.zfill(5)

                label_raw   = row.get("label")
                try: 
                    label   = float(label_raw)
                except (TypeError, ValueError) as e: 
                    raise ValueError(f"invalid label {label_raw} in {self.index_csv}") from e 

                byte_offset = int(row["byte_offset"])
                num_tiles   = int(row["num_tiles"])

                if byte_offset % self.dtype.itemsize != 0: 
                    raise ValueError("byte_offset is not aligned to dtype size")

                byte_offsets.append(byte_offset)
                counts.append(num_tiles)
                labels.append(label)
                fips.append(f)

        return (
            np.asarray(fips, dtype="U5"),
            np.asarray(labels, dtype=np.float32),
            np.asarray(byte_offsets, dtype=np.int64),
            np.asarray(counts, dtype=np.int64)
        )

    def validate_offsets(self): 
        image_file_elems = os.path.getsize(self.bin_path) // self.dtype.itemsize 
        stats_file_elems = os.path.getsize(self.stats_bin_path) // self.stats_dtype.itemsize 

        for i in range(len(self.labels)): 
            byte_offset = int(self.offset_bytes[i])
            n_tiles     = int(self.num_tiles[i])

            start_elem  = byte_offset // self.dtype.itemsize 
            end_elem    = start_elem + n_tiles * self.tile_elems 
            if end_elem > image_file_elems: 
                raise ValueError(f"index row {i} points past end of dataset.bin")

            tile_start  = byte_offset // self.image_bytes_per_tile 
            stats_end   = (tile_start + n_tiles) * self.stats_elems_per_tile 
            if stats_end > stats_file_elems: 
                raise ValueError(f"index row {i} points past end of stats.bin")

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
        y = np.asarray(labels, dtype=np.float32).reshape(-1)

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
        out = [self.rows[idx], float(self.labels[idx])]
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
    label: float  
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
        out_stats_path: str, 
        out_index_path: str, 
        patch_size: int = 32, 
        empty_threshold: float = 1.0, # fraction of zeros in mask
        stats_dtype=np.float32 
    ): 
        self.source          = source 
        self.out_bin_path    = out_bin_path 
        self.out_stats_path  = out_stats_path
        self.out_index_path  = out_index_path 
        self.patch_size      = patch_size 
        self.empty_threshold = empty_threshold
        self.stats_dtype     = np.dtype(stats_dtype)

    def write(self): 
        tile_shape = self.source.tile_shape 
        dtype      = np.dtype(self.source.dtype)

        _, h, w    = tile_shape 
        if h % self.patch_size != 0 or w % self.patch_size != 0: 
            raise ValueError("tile H/W must be divisible by patch_size for stats writing")

        bytes_per_tile = dtype.itemsize 
        for d in tile_shape: 
            bytes_per_tile *= int(d)

        os.makedirs(os.path.dirname(self.out_bin_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.out_stats_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.out_index_path), exist_ok=True)

        # For output print 
        t0        = time.perf_counter()
        last      = t0 
        bag_count = 0 

        with (open(self.out_bin_path, "wb") as bin_f, 
              open(self.out_stats_path, "wb") as stats_f,
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

                    stats = tile_patch_stats(tile, patch_size=self.patch_size)
                    stats = stats.astype(self.stats_dtype, copy=False)

                    bin_f.write(tile.tobytes(order="C"))
                    stats_f.write(stats.tobytes(order="C"))
                    n_written += 1 

                if n_written == 0: 
                    continue 

                writer.writerow({
                    "fips": bag.fips, 
                    "label": float(bag.label),
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
        tile_hw: tuple[int, int] = (256, 256),
        patch_size: int = 32 
    ): 
        if counties_path is None: 
            counties_path = project_path(
                "data", "geography", "county_shapefile", "tl_2020_us_county.shp"
            )

        self.counties_path = counties_path 
        self.tile_hw       = tuple(int(x) for x in tile_hw)
        self.patch_size    = patch_size  
        self.max_counties  = max_counties if max_counties is None else int(max_counties)

        if self.tile_hw[0] % self.patch_size != 0 or self.tile_hw[1] % self.patch_size != 0: 
            raise ValueError("tile_hw must be divisible by patch_size")
        
        if label_year == 2013: 
            self.label_map, _ = build_label_map(2013, census_dir=census_dir)
        else: 
            _, edges          = build_label_map(2013, census_dir=census_dir)
            self.label_map, _ = build_label_map(label_year, train_edges=edges, census_dir=census_dir)
    
    def save(
        self,
        *,
        out_bin_path: str, 
        out_stats_path: str, 
        out_index_path: str, 
        empty_threshold: float = 1.0
    ):
        '''
        Wraps BinaryTileWriter
        '''
        writer = BinaryTileWriter(
            self,
            out_bin_path=out_bin_path,
            out_stats_path=out_stats_path,
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
        patch_size: int = 32, 
        all_touched: bool = False, 
        log_scale: bool = True, 
        max_counties: int | None = None, 
        force_tight_crop: bool = False, 
        glcm_levels: int = 32, 
        glcm_distance: int = 1, 
        glcm_block: int = 32 
    ): 
        super().__init__(
            counties_path=counties_path,
            label_year=label_year,
            census_dir=census_dir, 
            max_counties=max_counties,
            tile_hw=tile_hw,
            patch_size=patch_size
        )
        self.viirs_path        = viirs_path 
        self.force_tight_crop  = bool(force_tight_crop)
        self.all_touched       = bool(all_touched)
        self.log_scale         = bool(log_scale)
        self.glcm_levels       = glcm_levels 
        self.glcm_distance     = glcm_distance 
        self.glcm_block        = glcm_block

    @property
    def tile_shape(self) -> tuple[int, int, int]:
        ht, wt = self.tile_hw 
        return (3, ht, wt)

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

                q = self.quantize_glcm(data, valid_mask, self.glcm_levels)
                glcm_contrast, glcm_entropy = self.glcm_block_maps(q, valid_mask)
                tile = np.stack([data, glcm_contrast, glcm_entropy], axis=0)
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

    def glcm_block_maps(self, q: NDArray, valid_mask: NDArray) -> tuple[NDArray, NDArray]: 
        h, w = q.shape 
        b    = self.glcm_block
        
        contrast_map = np.zeros((h, w), dtype=np.float32)
        entropy_map  = np.zeros((h, w), dtype=np.float32)

        for y0 in range(0, h, b): 
            for x0 in range(0, w, b): 
                y1, x1 = min(y0 + b, h), min(x0 + b, w)
                qb     = q[y0:y1, x0:x1]
                mb     = valid_mask[y0:y1, x0:x1]
                c, e   = self.glcm_metrics_block(qb, mb, self.glcm_levels, self.glcm_distance)
                contrast_map[y0:y1, x0:x1] = c 
                entropy_map[y0:y1, x0:x1]  = e 

        contrast_map[~valid_mask] = 0.0 
        entropy_map[~valid_mask]  = 0.0 
        return contrast_map, entropy_map

    @staticmethod 
    def glcm_metrics_block(
        q_block: NDArray,
        m_block: NDArray,
        levels: int, 
        distance: int 
    ) -> tuple[float, float]: 
        if q_block.size == 0 or not np.any(m_block): 
            return 0.0, 0.0 


        offsets = ((0, distance), (distance, 0), (distance, distance), (-distance, distance))
        
        P = np.zeros((levels, levels), dtype=np.float32)

        h, w = q_block.shape 
        for dy, dx in offsets: 
            y0, y1 = max(0, -dy), min(h, h - dy)
            x0, x1 = max(0, -dx), min(w, w - dx)
            if y1 <= y0 or x1 <= x0: 
                continue 

            a  = q_block[y0:y1, x0:x1]
            b  = q_block[y0+dy:y1+dy, x0+dx:x1+dx]
            ma = m_block[y0:y1, x0:x1]
            mb = m_block[y0+dy:y1+dy, x0+dx:x1+dx]
            pm = ma & mb 
            if not np.any(pm): 
                continue 

            idx = a[pm].astype(np.int64) * levels + b[pm].astype(np.int64)
            H   = np.bincount(idx, minlength=levels**2).reshape(levels, levels)
            P  += H + H.T  

        s = P.sum() 
        if s <= 0.0: 
            return 0.0, 0.0 

        P /= s 
        
        i = np.arange(levels, dtype=np.float32)[:, None]
        j = np.arange(levels, dtype=np.float32)[None, :]

        contrast_glcm  = float(((i - j)**2 * P).sum())
        entropy_glcm   = float(-(P * np.log(P + 1e-9)).sum())
        contrast_glcm /= float((levels - 1)**2 + 1e-9)
        entropy_glcm  /= float(np.log(levels**2) + 1e-9)
        return contrast_glcm, entropy_glcm


    @staticmethod
    def quantize_glcm(data: NDArray, valid_mask: NDArray, levels: int) -> NDArray: 
        q    = np.zeros_like(data, dtype=np.int64)
        vals = data[valid_mask]
        if vals.size == 0: 
            return q 
        hi = float(np.quantile(vals, 0.995))
        if hi <= 0.0: 
            return q 
        scaled = np.clip(data / hi, 0.0, 1.0)
        q      = np.floor(scaled * (levels - 1) + 1e-9).astype(np.int16)
        q[~valid_mask] = 0 
        return q 

# ---------------------------------------------------------
# Main entry point  
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--viirs-path", default=project_path(
        "data", "viirs", "viirs_2013_median_masked.tif"))

    parser.add_argument("--counties-path", default=project_path(
        "data", "geography", "county_shapefile", "tl_2020_us_county.shp"))

    parser.add_argument("--year", type=int, default=2013)
    parser.add_argument("--census-dir", default=project_path("data", "census"))
    parser.add_argument("--viirs-out", required=True)

    parser.add_argument("--no-log-scale", action="store_true")
    
    parser.add_argument("--viirs", action="store_true")

    args = parser.parse_args()

    if args.viirs:
        viirs_out = args.viirs_out
        bin_out   = Path(viirs_out) / "dataset.bin" 
        index_out = Path(viirs_out) / "index.csv"
        stats_out = Path(viirs_out) / "stats.bin"

        viirs = ViirsTensorDataset(
            viirs_path=args.viirs_path,
            counties_path=args.counties_path,
            label_year=args.year,
            census_dir=args.census_dir, 
            log_scale=not args.no_log_scale,
        )
        viirs.save(
            out_bin_path=str(bin_out), 
            out_stats_path=str(stats_out), 
            out_index_path=str(index_out)
        )


if __name__ == "__main__": 
    main() 
