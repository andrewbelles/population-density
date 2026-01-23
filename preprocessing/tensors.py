#!/usr/bin/env python3 
# 
# tensors.py  Andrew Belles  Jan 20th, 2026 
# 
# Image Tensor datasets representing Spatial Locations. Requires extra work to pack, tile, and 
# canvas counties together so that they are on uniform images for SpatialClassifier to use. 
# 

import imageio, fiona, csv, json, rasterio, rasterio.mask, argparse, sys, time, os

import numpy as np 

from pathlib import Path 

from typing import Tuple, Dict, Iterable 

from numpy.typing import NDArray

from abc import ABC, abstractmethod

from rasterio.warp import transform_geom 

from rasterio.errors import WindowError

from utils.helpers import project_path 

from utils.resources import LRUCache

# --------------------------------------------------------
# Tensor Loaders. Pre-packed and Lazy Loader  
# --------------------------------------------------------

class SpatialLazyLoader: 
    '''
    Loads saved spatial samples (VIIRS, NLCD, TIGER) and packs into ROI canvases.
    '''
    def __init__(self, root_dir: str): 
        self.is_packed = False 
        self.root_dir  = Path(root_dir)
        manifest       = self.root_dir / "manifest.jsonl"
        self.records   = [
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx): 
        rec = self.records[idx]
        with np.load(self.root_dir / rec["path"]) as npz: 
            spatial = npz["spatial"]
            mask    = npz["mask"]
        return spatial, mask, int(rec["label"]), rec["fips"]

    @staticmethod 
    def pack(batch, canvas_hw=(512, 512), tile_hw=None, weight_by_mask=True, gap: int = 32): 
        '''
        Greedy packing tool designed for ROI pooling, takes batches and aggressively packs 
        samples together on (H,W) images. 
        '''

        if tile_hw is None: 
            tile_hw = canvas_hw 
        
        H, W          = canvas_hw 
        canvases      = []
        rois          = [] # list of (canvas index, y0, x0, y1, x1)
        labels, fips  = [], []
        group_ids     = []
        group_weights = [] 

        for sample_idx, (spatial, mask, label, fip) in enumerate(batch): 
            labels.append(label)
            fips.append(fip)

            if spatial.ndim == 2: 
                spatial = spatial[None, ...]
            elif spatial.ndim != 3: 
                raise ValueError(f"spatial must be 2d/3d, got {spatial.shape}")

            if mask.ndim == 3: 
                mask = mask[0]
            elif mask.ndim != 2: 
                raise ValueError(f"mask must be 2d/3d, got {mask.shape}")

            for tile, tile_mask in SpatialLazyLoader._iter_tiles(spatial, mask, tile_hw): 
                h, w   = tile.shape[-2], tile.shape[-1] 
                weight = float(tile_mask.sum()) if weight_by_mask else 1.0 

                placed = False 

                # Try to place current bbox on any active canvases, if not make a new one 
                for c_idx, (canvas, mask_canvas, cursor) in enumerate(canvases): 
                    y, x, row_h = cursor 

                    if x + w > W: 
                        y     = y + row_h + gap 
                        x     = 0 
                        row_h = 0

                    if y + h <= H and x + w <= W: 

                        canvas[:, y:y+h, x:x+w]      = tile 
                        mask_canvas[0, y:y+h, x:x+w] = tile_mask 

                        rois.append((c_idx, y, x, y + h, x + w))

                        group_ids.append(sample_idx)
                        group_weights.append(weight)

                        cursor[:] = [y, x + w + gap, max(row_h, h)]   
                        placed = True 
                        break 

                    new_y = y + h + gap 
                    if new_y + h <= H: 
                        
                        cursor[0] = new_y 
                        cursor[1] = 0 
                        
                        canvas[:, new_y:new_y+h, 0:w] = tile 
                        mask_canvas[0, new_y:new_y+h, 0:w] = tile_mask 
                        rois.append((c_idx, new_y, 0, new_y + h, w))

                        group_ids.append(sample_idx)
                        group_weights.append(weight)

                        cursor[1] = w + gap 
                        placed = True 
                        break 

                if not placed: 

                    canvas      = np.zeros((spatial.shape[0], H, W), dtype=spatial.dtype)
                    mask_canvas = np.zeros((1, H, W), dtype=np.uint8)

                    canvas[:, :h, :w]      = tile 
                    mask_canvas[0, :h, :w] = tile_mask

                    canvases.append([canvas, mask_canvas, [0, w + gap, h]])
                    rois.append((len(canvases) - 1, 0, 0, h, w)) 
                    group_ids.append(sample_idx)
                    group_weights.append(weight)

        if not canvases: 
            raise ValueError("pack() produced no canvases")

        packed_images = np.stack([c[0] for c in canvases], axis=0)
        packed_masks  = np.stack([c[1] for c in canvases], axis=0)
        return (packed_images, packed_masks, rois, np.asarray(labels), 
                fips, group_ids, group_weights)

    @staticmethod 
    def _iter_tiles(spatial, mask, tile_hw): 
        H, W = tile_hw 
        h, w = spatial.shape[-2:]
        for y0 in range(0, h, H): 
            for x0 in range(0, w, W): 
                y1 = min(y0 + H, h)
                x1 = min(x0 + W, w)
                tile      = spatial[:, y0:y1, x0:x1] 
                tile_mask = mask[y0:y1, x0:x1]
                if tile_mask.sum() == 0: 
                    continue 
                yield tile, tile_mask 


class SpatialPackedLoader: 
    '''
    Loads pre-packed canvases created by SpatialTensorDataset.save_packed. 
    Each item contains full batch-ready tensors and ROI metadata 
    '''

    def __init__(self, root_dir: str, *, cache_mb: int = 0, cache_items: int = 0): 
        self.is_packed = True 
        self.root_dir  = Path(root_dir)
        manifest       = self.root_dir / "manifest.jsonl"
        self.records   = [
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() 
            if line.strip() 
        ]

        self.cache = None 
        if cache_mb > 0 or cache_items > 0: 
            self.cache = LRUCache(
                max_bytes=(cache_mb * 1024 * 1024) if cache_mb > 0 else None, 
                max_items=(cache_items if cache_items > 0 else None)
            )

        self.prefetch_workers = 0 
        self.prefetch_factor  = 4

        # cache metadata 
        self._cache_reads  = 0 
        self._cache_hits   = 0 
        self._cache_misses = 0

    def __len__(self): 
        return len(self.records)

    def __getitem__(self, idx): 
        rec = self.records[idx]
        key = rec["path"]

        self._cache_reads += 1 
        if self.cache is not None and self._cache_reads % 1000 == 0: 
            hit_rate     = self._cache_hits / max(self._cache_reads, 1)
            cached_items = len(self.cache.data)
            cached_mb    = (self.cache.total / (1024**2))
            print(f"[cache] reads={self._cache_reads}, hit_rate={hit_rate:.2f}% "
                  f"items={cached_items} cache_mb={cached_mb:.1f}", file=sys.stderr)

        if self.cache is not None: 
            hit = self.cache.get(key)
            if hit is not None: 
                self._cache_hits += 1 
                canvases, masks, rois, labels, fips, group_ids, group_weights = hit 
                return (
                    canvases, 
                    masks, 
                    rois.tolist(),
                    labels,
                    list(fips),
                    group_ids, 
                    group_weights 
                )
            else: 
                self._cache_misses += 1 

        with np.load(self.root_dir / rec["path"]) as npz: 
            canvases      = npz["canvases"]
            masks         = npz["masks"]
            rois          = npz["rois"]
            labels        = npz["labels"]
            fips          = npz["fips"]
            group_ids     = npz["group_ids"]
            group_weights = npz["group_weights"]

        if self.cache is not None: 
            self.cache.put(key, (canvases, masks, rois, labels, fips, group_ids, group_weights))

        return (
            canvases, 
            masks, 
            rois.tolist(),
            labels,
            list(fips),
            group_ids, 
            group_weights 
        )

# ---------------------------------------------------------
# Tensor Dataset Parent Class 
# ---------------------------------------------------------

class SpatialTensorDataset(ABC): 

    '''
    Parent Class for spatial tensor datasets (VIIRS, NLCD, TIGER).
    Handles label loading, county iteration, and persistence. 
    '''

    def __init__(
        self,
        raster_path: str | None = None, 
        counties_path: str | None = None, 
        labels_path: str | None = None, 
        *,
        canvas_size: Tuple[int, int] | None = None, 
        max_counties: int | None = None, 
        debug_png_dir: str | None = None, 
        tight_crop: bool = True 
    ): 
        if counties_path is None: 
            counties_path = project_path(
                "data", "geography", "county_shapefile", "tl_2020_us_county.shp"
            )
        if labels_path is None: 
            labels_path = project_path("data", "nchs", "nchs_classification.csv")

        self.raster_path   = raster_path 
        self.counties_path = counties_path 
        self.labels_path   = labels_path 
        self.canvas_size   = canvas_size
        self.max_counties  = max_counties if max_counties is None else int(max_counties)
        self.debug_png_dir = debug_png_dir 
        self.tight_crop    = bool(tight_crop)
        self.label_map     = self._load_labels()

    def save(self, output_dir: str): 
        out_dir    = Path(output_dir)
        sample_dir = out_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        records = []
        t0      = time.perf_counter() 
        last    = t0
        count   = 0 
        for rec in self._iter_samples(): 
            fips    = rec["fips"]
            label   = int(rec["label"])
            spatial = rec["spatial"]
            mask    = rec["mask"]

            if self.tight_crop: 
                spatial, mask = self._tight_crop(spatial, mask)

            if spatial.size == 0:
                continue 

            if np.issubdtype(spatial.dtype, np.floating): 
                spatial_out = spatial.astype(np.float16, copy=False) 
            else: 
                spatial_out = spatial 

            out_path = sample_dir / f"fips_{fips}.npz"
            np.savez_compressed(
                out_path,
                spatial=spatial_out,
                mask=mask.astype(np.uint8),
                label=np.int64(label),
                fips=fips
            )

            records.append({
                "fips": fips,
                "label": label,
                "path": str(out_path.relative_to(out_dir)),
                "shape": [int(x) for x in spatial.shape],
                "mask_sum": int(mask.sum())
            })

            if self.debug_png_dir: 
                self._write_debug_png(fips, spatial, mask)

            count += 1 
            if count % 250 == 0: 
                now = time.perf_counter()
                print(
                    f"[save] {count} samples | +{now - last:.2f}s | total {now - t0:.2f}s",
                    file=sys.stderr,
                    flush=True
                )
                last = now 
            if self.max_counties is not None and count >= self.max_counties: 
                break 

        manifest_path = out_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f: 
            for r in records: 
                f.write(json.dumps(r) + "\n")

        print(f"[save] wrote {len(records)} samples to {out_dir} (manifest: {manifest_path})")

    def save_packed(
        self,
        out_dir, 
        pack_batch: int = 8,
        canvas_hw=(512, 512),
        tile_hw=None,
        gap: int = 32,
        weight_by_mask: bool = True, 
        pack_ram_mb: int | None = None, 
        max_canvases: int | None = 8
    ):
        out_dir    = Path(out_dir)
        packed_dir = out_dir / "packed" 
        packed_dir.mkdir(parents=True, exist_ok=True)

        if pack_ram_mb is None: 
            env_mb      = os.environ.get("TOPG_TENSOR_RAM_MB", "")
            pack_ram_mb = int(env_mb) if env_mb.strip() else None 

        ram_bytes = None 
        if pack_ram_mb is not None and pack_ram_mb > 0: 
            ram_bytes = pack_ram_mb * 1024 * 1024 

        pack_batch_eff   = int(pack_batch)
        max_canvases_eff = max_canvases
        
        def _apply_ram_limits(spatial, mask): 
            nonlocal pack_batch_eff, max_canvases_eff 
            if ram_bytes is None: 
                return 
            sample_bytes = spatial.nbytes + mask.nbytes 
            if sample_bytes > 0: 
                est_batch      = max(1, int((ram_bytes * 0.5) / sample_bytes))
                pack_batch_eff = max(1, min(pack_batch_eff, est_batch))

            bytes_per_canvas = self._estimate_canvas_bytes(spatial, canvas_hw)
            if bytes_per_canvas > 0: 
                est_canvases = max(1, int((ram_bytes * 0.5) / bytes_per_canvas))
                if max_canvases_eff is None: 
                    max_canvases_eff = est_canvases 
                else: 
                    max_canvases_eff = min(max_canvases_eff, est_canvases)

        batch          = []
        records        = []
        all_labels     = []
        all_fips       = []
        all_groups     = []
        pack_labels    = []
        pack_n_samples = []
        pack_idx       = 0
        count          = 0 

        t0   = time.perf_counter() 
        last = t0

        def _record_pack(batch, pack_idx):
            pack_y = [b[2] for b in batch]
            pack_f = [b[3] for b in batch]

            uniq, counts = np.unique(pack_y, return_counts=True)
            pack_labels.append(int(uniq[counts.argmax()]))
            pack_n_samples.append(len(pack_y))
            all_labels.extend(pack_y)
            all_fips.extend(pack_f)
            all_groups.extend([pack_idx] * len(pack_y))

        def _flush_capped(batch):
            nonlocal pack_idx
            if not batch: 
                return []
            batch, overflow = self._pack_with_cap(
                batch,
                max_canvases=max_canvases,
                canvas_hw=canvas_hw,
                tile_hw=tile_hw,
                gap=gap,
                weight_by_mask=weight_by_mask,
            )

            _record_pack(batch, pack_idx)
            meta = self._flush_pack(
                batch, pack_idx, out_dir, 
                canvas_hw=canvas_hw,
                tile_hw=tile_hw,
                gap=gap,
                weight_by_mask=weight_by_mask
            )
            records.append(meta)
            pack_idx += 1 
            return overflow 

        for rec in self._iter_samples(): 
            spatial, mask, label, fips = (rec["spatial"], rec["mask"], 
                                          int(rec["label"]), rec["fips"])

            if self.tight_crop: 
                spatial, mask = self._tight_crop(spatial, mask)

            if spatial.size == 0: 
                continue 

            _apply_ram_limits(spatial, mask)

            batch.append((spatial, mask, label, fips))
            count += 1 

            if len(batch) >= pack_batch: 
                overflow = _flush_capped(batch)
                batch[:] = overflow 

            if count % 250 == 0: 
                now = time.perf_counter()
                print(
                    f"[save] {count} samples | +{now - last:.2f}s | total {now - t0:.2f}s",
                    file=sys.stderr,
                    flush=True
                )
                last = now 

            if self.max_counties is not None and count >= self.max_counties: 
                break 

        overflow = _flush_capped(batch)
        while overflow: 
            overflow = _flush_capped(overflow)

        meta_path = out_dir / "packed" / "meta.npz" 
        np.savez(
            meta_path,
            sample_labels=np.asarray(all_labels, dtype=np.int64),
            sample_ids_full=np.asarray(all_fips, dtype="U5"),
            sample_groups=np.asarray(all_groups, dtype=np.int64),
            pack_labels=np.asarray(pack_labels, dtype=np.int64),
            pack_n_samples=np.asarray(pack_n_samples, dtype=np.int64)
        )
        print(f"[save_packed] wrote meta: {meta_path}")

        manifest_path = out_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f: 
            for r in records: 
                f.write(json.dumps(r) + "\n")

        print(f"[save_packed] wrote {len(records)} packs to {out_dir} "
              f"(manifest: {manifest_path})")

    def _load_labels(self) -> Dict[str, int]: 
        path = Path(self.labels_path)
        if not path.exists(): 
            raise FileNotFoundError(f"label CSV not found: {path}")

        labels: Dict[str, int] = {}
        with path.open(newline="", encoding="utf-8") as f: 
            reader = csv.DictReader(f)
            if reader.fieldnames is None: 
                raise ValueError("label CSV missing header row")
            if "FIPS" not in reader.fieldnames or "class_code" not in reader.fieldnames: 
                raise ValueError(f"expected header FIPS,class_code, got {reader.fieldnames}")

            for row in reader: 
                fips = (row.get("FIPS") or "").strip() 
                code = (row.get("class_code") or "").strip() 
                if not fips or not code: 
                    continue 
                if fips.isdigit(): 
                    fips = fips.zfill(5)
                if code.isdigit():
                    labels[fips] = int(code) - 1 

        if not labels: 
            raise ValueError("label map is empty")
        return labels 

    def _iter_counties(
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

    def _pack_with_cap(
        self,
        batch,
        *,
        max_canvases: int | None, 
        canvas_hw, 
        tile_hw, 
        gap,
        weight_by_mask
    ): 
        if max_canvases is None: 
            return batch, None 

        overflow = []
        while len(batch) > 1: 
            canvases = SpatialLazyLoader.pack(
                batch, 
                canvas_hw=canvas_hw,
                tile_hw=tile_hw,
                gap=gap,
                weight_by_mask=weight_by_mask
            )[0]

            if canvases.shape[0] <= max_canvases:
                break 
            overflow.append(batch.pop()) 

        if len(batch) == 1:
            canvases = SpatialLazyLoader.pack(
                batch,
                canvas_hw=canvas_hw,
                tile_hw=tile_hw,
                gap=gap,
                weight_by_mask=weight_by_mask,
            )[0]

        overflow.reverse()
        return batch, overflow

    @abstractmethod
    def _iter_samples(self):
        '''
        Yield dicts with: fips, label, spatial, mask
        '''
        raise NotImplementedError

    def _write_debug_png(self, fips, spatial, mask): 
        if self.debug_png_dir is None: 
            return 
        out_dir = Path(self.debug_png_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        spatial_u8 = self._to_uint8(self._as_2d(spatial))
        mask_u8    = (self._as_2d(mask).astype(np.uint8) * 255)

        imageio.imwrite(out_dir / f"fips_{fips}_spatial.png", spatial_u8)
        imageio.imwrite(out_dir / f"fips_{fips}_mask.png", mask_u8)

    @staticmethod 
    def _flush_pack(
        batch,
        pack_idx: int,
        out_dir: Path,
        *,
        canvas_hw=(512, 512),
        tile_hw=None,
        gap: int = 32, 
        weight_by_mask: bool = True,
        compress: bool = True
    ) -> dict: 

        packed = SpatialLazyLoader.pack(
            batch, 
            canvas_hw=canvas_hw,
            tile_hw=tile_hw,
            gap=gap,
            weight_by_mask=weight_by_mask
        )
        canvases, masks, rois, labels, fips, group_ids, group_weights = packed

        out_path = out_dir / "packed" / f"pack_{pack_idx:05d}.npz"

        if np.issubdtype(canvases.dtype, np.floating): 
            canvases_out = canvases.astype(np.float16, copy=False) 
        else: 
            canvases_out = canvases

        save_fn = np.savez_compressed if compress else np.savez 
        save_fn(
            out_path,
            canvases=canvases_out,
            masks=masks.astype(np.uint8),
            rois=np.asarray(rois, dtype=np.int32),
            labels=np.asarray(labels, dtype=np.int64),
            fips=np.asarray(fips),
            group_ids=np.asarray(group_ids, dtype=np.int64),
            group_weights=np.asarray(group_weights, dtype=np.float32)
        )

        return {
            "path": str(out_path.relative_to(out_dir)),
            "n_samples": int(len(labels)),
            "n_canvases": int(canvases.shape[0]),
            "n_rois": int(len(rois))
        }


    @staticmethod 
    def _tight_crop(data, mask): 
        ys, xs = np.nonzero(mask)
        if ys.size == 0: 
            if data.ndim == 3: 
                return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.uint8)
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.uint8)

        y0, y1 = ys.min(), ys.max() + 1 
        x0, x1 = xs.min(), xs.max() + 1 

        if data.ndim == 2: 
            crop_data = data[y0:y1, x0:x1]
        elif data.ndim == 3: 
            crop_data = data[:, y0:y1, x0:x1]
        else: 
            raise ValueError(f"expected 2d/3d data, got shape {data.shape}")

        crop_mask = mask[y0:y1, x0:x1].astype(np.uint8)
        return crop_data.astype(np.float32), crop_mask 

    @staticmethod 
    def _as_2d(x): 
        if x.ndim == 2: 
            return x 
        if x.ndim == 3: 
            return x.mean(axis=0)
        raise ValueError(f"expected 2d/3d, got shape {x.shape}")

    @staticmethod 
    def _to_uint8(x, vmin=None, vmax=None): 
        if vmin is None: 
            vmin = float(np.nanmin(x))
        if vmax is None: 
            vmax = float(np.nanmax(x))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin: 
            return np.zeros_like(x, dtype=np.uint8)
        y = (x - vmin) / (vmax - vmin)
        y = np.clip(y, 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)

    @staticmethod 
    def _estimate_canvas_bytes(spatial, canvas_hw): 
        H, W       = canvas_hw 
        channels   = spatial.shape[0] if spatial.ndim == 3 else 1 
        data_bytes = channels * H * W * spatial.dtype.itemsize 
        mask_bytes = H * W 
        return data_bytes + mask_bytes

# ---------------------------------------------------------
# Viirs nighttime lights tensor dataset  
# ---------------------------------------------------------

class ViirsTensorDataset(SpatialTensorDataset):
    '''
    VIIRS tensor generator (single channel) 
    '''

    def __init__(
        self,
        viirs_path: str | None = None, 
        counties_path: str | None = None, 
        labels_path: str | None = None,
        *,
        all_touched: bool = False, 
        log_scale: bool = True, 
        max_counties: int | None = None, 
        debug_png_dir: str | None = None 
    ): 
        super().__init__(
            raster_path=viirs_path,
            counties_path=counties_path,
            labels_path=labels_path,
            max_counties=max_counties,
            debug_png_dir=debug_png_dir,
            tight_crop=True
        )
        self.all_touched = bool(all_touched)
        self.log_scale   = bool(log_scale)

    def _iter_samples(self):
        if self.raster_path is None: 
            raise ValueError("viirs_path is required")

        with rasterio.open(self.raster_path) as src: 
            nodata = src.nodata 
            with fiona.open(self.counties_path) as shp: 
                for fips, geom in self._iter_counties(shp, src.crs): 
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
                    if nodata is not None: 
                        valid_mask &= arr != nodata 
                    if not np.any(valid_mask): 
                        continue 

                    data = np.asarray(arr, dtype=np.float32)
                    if self.log_scale: 
                        data = np.log1p(np.maximum(data, 0.0))
                    data[~valid_mask] = 0.0 

                    yield {
                        "fips": fips,
                        "label": self.label_map[fips],
                        "spatial": data, 
                        "mask": valid_mask.astype(np.uint8)
                    }

# ---------------------------------------------------------
# National land usage tensor dataset  
# ---------------------------------------------------------

class NlcdTensorDataset(SpatialTensorDataset): 
    '''
    NLCD multi-channel categorical tensor generator 
    channel_groups: dict[str, list[int]] mapping NLCD codes to channels 
    '''

    def __init__(
        self,
        nlcd_path: str | None = None, 
        counties_path: str | None = None, 
        labels_path: str | None = None, 
        *,
        channel_groups: dict[str, list[int]] | None = None, 
        all_touched: bool = False, 
        max_counties: int | None = None,
        downsample: int = 1, 
        debug_png_dir: str | None = None 
    ): 
        super().__init__(
            raster_path=nlcd_path,
            counties_path=counties_path,
            labels_path=labels_path,
            max_counties=max_counties,
            debug_png_dir=debug_png_dir,
            tight_crop=True
        )
        self.all_touched = bool(all_touched)

        if channel_groups is None: 
            channel_groups = {
                "water": [11, 12],
                "developed": [21, 22, 23, 24],
                "forest": [41, 42, 43],
                "shrub": [52],
                "grass": [71, 72, 73, 74],
                "cropland": [81, 82],
                "wetlands": [90, 95]
            }
        self.channel_groups = channel_groups 
        self.downsample     = downsample
        self._lut           = self._build_lut(self.channel_groups)

    def _build_lut(self, groups: dict[str, list[int]]) -> dict[int, int]: 
        lut = {}
        for idx, codes in enumerate(groups.values()): 
            for c in codes: 
                lut[int(c)] = idx + 1 
        return lut 

    def _iter_samples(self): 
        if self.raster_path is None: 
            raise ValueError("nlcd_path is required")

        with rasterio.open(self.raster_path) as src: 
            nodata = src.nodata 
            with fiona.open(self.counties_path) as shp: 
                for fips, geom in self._iter_counties(shp, src.crs): 
                    try: 
                        out_image, _ = rasterio.mask.mask(
                            src, [geom], crop=True, filled=False, all_touched=self.all_touched
                        )
                    # if geometry does not overlap w/ raster bounds, just move on
                    except (ValueError, WindowError):
                        print("[nlcd window error] skipping sample...", file=sys.stderr)
                        continue 
                    arr = out_image[0]
                    if arr.size == 0: 
                        continue 

                    valid_mask = ~arr.mask if hasattr(arr, "mask") else np.isfinite(arr)
                    if nodata is not None: 
                        valid_mask &= arr != nodata 
                    if not np.any(valid_mask): 
                        continue 

                    data = np.asarray(arr, dtype=np.float32)
                    data[~valid_mask] = -1 

                    H, W = data.shape 
                    spatial = np.zeros((H, W), dtype=np.uint8)
                    for code, ch in self._lut.items():
                        spatial[data == code] = ch

                    if self.downsample > 1: 
                        spatial, valid_mask = self._block_mode_downsample(
                            spatial, valid_mask, self.downsample 
                        )

                    yield {
                        "fips": fips,
                        "label": self.label_map[fips],
                        "spatial": spatial,
                        "mask": valid_mask.astype(np.uint8)
                    }

    @staticmethod 
    def _block_mode_downsample(spatial: NDArray, mask: NDArray, factor: int): 
        if factor <= 1: 
            return spatial, mask 

        H, W = spatial.shape 
        h    = (H // factor) * factor 
        w    = (W // factor) * factor 
        
        spatial = spatial[:h, :w]
        mask    = mask[:h, :w].astype(bool)

        hf = h // factor 
        wf = w // factor 

        s = spatial.reshape(hf, factor, wf, factor)
        m = mask.reshape(hf, factor, wf, factor) 

        s = s.reshape(hf, wf, factor * factor)
        m = m.reshape(hf, wf, factor * factor) 

        s = np.where(m, s, -1)

        n_classes = int(spatial.max()) + 1 
        counts    = np.zeros((hf, wf, n_classes), dtype=np.int32)
        for c in range(n_classes): 
            counts[..., c] = np.sum(s == c, axis=2)

        out      = counts.argmax(axis=2).astype(np.uint8)
        out_mask = (m.any(axis=2)).astype(np.uint8)
        return out, out_mask

# ---------------------------------------------------------
# Main entry point  
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--viirs-path", default=project_path(
        "data", "viirs", "viirs_2023_avg_masked.dat.tif"))
    parser.add_argument("--nlcd-path", default=project_path(
        "data", "nlcd", "Annual_NLCD_LndCov_2023_CU_C1V1.tif"))
    parser.add_argument("--counties-path", default=project_path(
        "data", "geography", "county_shapefile", "tl_2020_us_county.shp"))
    parser.add_argument("--labels-path", default=project_path(
        "data", "nchs", "nchs_classification.csv"))
    parser.add_argument("--out-root", default=project_path("data", "tensors"))
    parser.add_argument("--viirs-out", default=None)
    parser.add_argument("--nlcd-out", default=None)

    parser.add_argument("--debug-png-dir", default=None)
    parser.add_argument("--no-log-scale", action="store_true")
    parser.add_argument("--skip-viirs", action="store_true")
    parser.add_argument("--skip-nlcd", action="store_true")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--pack-size", type=int, default=8)
    parser.add_argument("--pack-ram-mb", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args()

    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    viirs_out = args.viirs_out or str(out_root / "viirs_roi")
    nlcd_out  = args.nlcd_out  or str(out_root / "nlcd_roi")

    if not args.skip_viirs:
        viirs = ViirsTensorDataset(
            viirs_path=args.viirs_path,
            counties_path=args.counties_path,
            labels_path=args.labels_path,
            log_scale=not args.no_log_scale,
            debug_png_dir=args.debug_png_dir
        )
        viirs.save(viirs_out) if not args.packed else viirs.save_packed(
            viirs_out,
            pack_batch=args.pack_size,
            pack_ram_mb=args.pack_ram_mb
        )

    if not args.skip_nlcd: 
        nlcd = NlcdTensorDataset(
            nlcd_path=args.nlcd_path,
            counties_path=args.counties_path,
            labels_path=args.labels_path,
            debug_png_dir=args.debug_png_dir,
            downsample=args.downsample
        )
        nlcd.save(nlcd_out) if not args.packed else nlcd.save_packed(
            nlcd_out,
            pack_batch=args.pack_size,
            pack_ram_mb=args.pack_ram_mb
        )


if __name__ == "__main__": 
    main() 
