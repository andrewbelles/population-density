#!/usr/bin/env python3 
# 
# ..  Andrew Belles  Jan 14th, 2026 
# 
# 
# 
# 

import argparse, csv, fiona, rasterio, imageio
import json 

from pathlib import Path 

from typing import Dict, Iterable, Tuple 

import numpy as np 

import rasterio.mask 

from rasterio.warp import transform_geom 

from utils.helpers import project_path


class ViirsTensorDataset: 

    def __init__(
        self,
        viirs_path: str | None = None, 
        counties_path: str | None = None, 
        labels_path: str | None = None, 
        *,
        canvas_h: int = 128, 
        canvas_w: int = 128, 
        gaf_size: int = 64,
        all_touched: bool = False, 
        log_scale: bool = False, 
        max_counties: int | None = None,
        debug_png_dir: str | None = None
    ): 

        if viirs_path is None: 
            viirs_path = project_path("data", "viirs", "viirs_2023_avg_masked.dat.tif")
        if counties_path is None: 
            counties_path = project_path("data", "geography", "county_shapefile", 
                                         "tl_2020_us_county.shp")
        if labels_path is None: 
            labels_path = project_path("data", "nchs", "nchs_classification.csv")

        self.viirs_path    = viirs_path 
        self.counties_path = counties_path 
        self.labels_path   = labels_path 
        self.canvas_h      = int(canvas_h)
        self.canvas_w      = int(canvas_w)
        self.gaf_size      = int(gaf_size)
        self.all_touched   = bool(all_touched)
        self.log_scale     = bool(log_scale)
        self.max_counties  = max_counties if max_counties is None else int(max_counties)
        self.debug_png_dir = debug_png_dir

        self.label_map     = self._load_labels()

    def save(self, output_dir: str): 
        out_dir    = Path(output_dir)
        sample_dir = out_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        records = []
        count   = 0 
        for rec in self._iter_samples(): 
            fips     = rec["fips"]
            out_path = sample_dir / f"fips_{fips}.npz"

            np.savez_compressed(
                out_path, 
                spatial=rec["spatial"].astype(np.float32),
                mask=rec["mask"].astype(np.uint8),
                gaf=rec["gaf"].astype(np.float32),
                label=np.int64(rec["label"]),
                fips=fips
            )
            records.append({
                "fips": fips, 
                "label": int(rec["label"]),
                "path": str(out_path.relative_to(out_dir)),
                "h": int(rec["spatial"].shape[0]),
                "w": int(rec["spatial"].shape[1]),
                "mask_sum": int(rec["mask"].sum())
            })
            count += 1 
            if self.max_counties is not None and count >= self.max_counties: 
                break 

        manifest_path = out_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f: 
            for r in records: 
                f.write(json.dumps(r) + "\n")

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

    def _iter_samples(self): 
        viirs_path    = Path(self.viirs_path)
        counties_path = Path(self.counties_path)
        if not viirs_path.exists(): 
            raise FileNotFoundError(f"VIIRS raster not found: {viirs_path}")
        if not counties_path.exists(): 
            raise FileNotFoundError(f"county shapefile not found: {counties_path}")

        with rasterio.open(viirs_path) as src: 
            nodata = src.nodata 
            with fiona.open(counties_path) as shp: 
                for fips, geom in self._iter_counties(shp, src.crs): 
                    out_image, _ = rasterio.mask.mask(
                        src, [geom], crop=True, filled=False, all_touched=self.all_touched
                    )
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

                    spatial, mask = self._tight_crop(data, valid_mask)
                    if spatial.size == 0: 
                        continue 

                    gaf = self._gaf_from_mask(spatial, mask.astype(bool), self.gaf_size)

                    if self.debug_png_dir: 
                        self._write_debug_png(fips, spatial, mask, gaf)

                    yield {
                        "fips": fips, 
                        "label": self.label_map[fips],
                        "spatial": spatial,
                        "mask": mask,
                        "gaf": gaf
                    }

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

    def _write_debug_png(self, fips, spatial, mask, gaf): 
        if not self.debug_png_dir: 
            return 
        out_dir = Path(self.debug_png_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        spatial_u8 = self._to_uint8(self._as_2d(spatial))
        mask_u8    = (self._as_2d(mask).astype(np.uint8) * 255)
        gaf_u8     = self._to_uint8(self._as_2d(gaf), vmin=-1.0, vmax=1.0)

        imageio.imwrite(out_dir / f"fips_{fips}_spatial.png", spatial_u8)
        imageio.imwrite(out_dir / f"fips_{fips}_mask.png", mask_u8)
        imageio.imwrite(out_dir / f"fips_{fips}_gaf.png", gaf_u8)

    @staticmethod 
    def _tight_crop(data, mask): 
        ys, xs = np.nonzero(mask)
        if ys.size == 0: 
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.uint8)

        y0, y1 = ys.min(), ys.max() + 1 
        x0, x1 = xs.min(), xs.max() + 1 
        
        crop_data = data[y0:y1, x0:x1].astype(np.float32)
        crop_mask = mask[y0:y1, x0:x1].astype(np.uint8)
        return crop_data, crop_mask

    @staticmethod 
    def _gaf_from_mask(data, valid_mask, size):
        '''
        DFT conversion of spatial image into frequency representation 
        '''

        seq  = data[valid_mask].astype(np.float32).ravel() 
        seq  = ViirsTensorDataset._resample_1d(seq, size)
        vmin = float(seq.min())
        vmax = float(seq.max())
        if vmax <= vmin:
            return np.zeros((size, size), dtype=np.float32)

        scaled = (seq - vmin) / (vmax - vmin)
        scaled = scaled * 2.0 - 1.0 
        scaled = np.clip(scaled, -1.0, 1.0)

        phi    = np.arccos(scaled.astype(np.float64))
        gaf    = np.cos(phi[:, None] + phi[None, :])
        return gaf.astype(np.float32)

    @staticmethod 
    def _resample_1d(seq, n): 
        if seq.size == 0: 
            return np.zeros(n, dtype=np.float32)
        if seq.size == n: 
            return seq.astype(np.float32)

        x_old = np.linspace(0.0, 1.0, num=seq.size, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=n, dtype=np.float64)
        out   = np.interp(x_new, x_old, seq.astype(np.float64))
        return out.astype(np.float32)
    
    def _as_2d(self, x): 
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] == 1: 
            x = x[0]
        elif x.ndim == 3 and x.shape[-1] == 1: 
            x = x[..., 0]
        if x.ndim != 2: 
            raise ValueError(f"expected 2d image, got shape {x.shape}")
        return x 

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


class ViirsLazyLoader: 
    '''
    Lazy Loader that pulls a single county on __getitem__ instead of entire dataset via 
    dataset manifest saved by ViirsTensorDataset
    '''
    def __init__(self, root_dir: str): 
        self.root_dir = Path(root_dir)
        manifest      = self.root_dir / "manifest.jsonl"
        self.records  = [
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if 
            line.strip()
        ]

    def __len__(self): 
        return len(self.records)

    def __getitem__(self, idx): 
        rec = self.records[idx] 
        with np.load(self.root_dir / rec["path"]) as npz: 
            spatial = npz["spatial"]
            mask    = npz["mask"]
            gaf     = npz["gaf"]
        return spatial, mask, gaf, int(rec["label"]), rec["fips"]

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

        for sample_idx, (spatial, mask, _, label, fip) in enumerate(batch): 
            labels.append(label)
            fips.append(fip)

            for tile, tile_mask in ViirsLazyLoader._iter_tiles(spatial, mask, tile_hw): 
                h, w   = tile.shape 
                weight = float(tile_mask.sum()) if weight_by_mask else 1.0 

                placed = False 
                # Try to place current bbox on any active canvases, if not make a new one 
                for c_idx, (canvas, mask_canvas, cursor) in enumerate(canvases): 
                    y, x = cursor 
                    if y + h <= H and x + w <= W: 
                        canvas[0, y:y+h, x:x+w]      = tile 
                        mask_canvas[0, y:y+h, x:x+w] = tile_mask 
                        rois.append((c_idx, y, x, y + h, x + w))
                        group_ids.append(sample_idx)
                        group_weights.append(weight)
                        cursor[1] = x + w + gap 
                        placed = True 
                        break 
                if not placed: 
                    canvas      = np.zeros((1, H, W), dtype=np.float32)
                    mask_canvas = np.zeros((1, H, W), dtype=np.uint8)
                    canvas[0, :h, :w]      = tile 
                    mask_canvas[0, :h, :w] = tile_mask
                    canvases.append([canvas, mask_canvas, [0, w + gap]])
                    rois.append((len(canvases) - 1, 0, 0, h, w)) 
                    group_ids.append(sample_idx)
                    group_weights.append(weight)


        packed_images = np.stack([c[0] for c in canvases], axis=0)
        packed_masks  = np.stack([c[1] for c in canvases], axis=0)
        return (packed_images, packed_masks, rois, np.asarray(labels), 
                fips, group_ids, group_weights)

    @staticmethod 
    def _iter_tiles(spatial, mask, tile_hw): 
        H, W = tile_hw 
        h, w = spatial.shape 
        for y0 in range(0, h, H): 
            for x0 in range(0, w, W): 
                tile      = spatial[y0:y0 + H, x0:x0 + W] 
                tile_mask = mask[y0:y0 + H, x0:x0 + W]
                if tile_mask.sum() == 0: 
                    continue 
                yield tile, tile_mask 

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--out", default=project_path("data", "datasets"))
    parser.add_argument("--all-touched", action="store_true")
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--debug-png-dir", default=None)
    args = parser.parse_args()

    data = ViirsTensorDataset(
        all_touched=args.all_touched,
        log_scale=args.log_scale,
        debug_png_dir=args.debug_png_dir
    )
    data.save(args.out)


if __name__ == "__main__": 
    main() 
