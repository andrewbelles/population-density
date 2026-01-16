#!/usr/bin/env python3 
# 
# ..  Andrew Belles  Jan 14th, 2026 
# 
# 
# 
# 

import argparse, csv, fiona, rasterio, imageio 

from pathlib import Path 

from typing import Dict, Iterable, Tuple 

import numpy as np 

import rasterio.mask 

from rasterio.warp import transform_geom 

from utils.helpers import project_path

from scipy.io      import savemat 
from scipy.ndimage import zoom 


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
        self.data          = self._build() 

    def save(self, output_path: str): 
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        savemat(out, self.data)
        n = self.data["labels"].shape[0]
        print(f"[viirs tensor] saved {out} ({n} counties)")

    def _build(self): 
        viirs_path    = Path(self.viirs_path)
        counties_path = Path(self.counties_path)
        if not viirs_path.exists():
            raise FileNotFoundError(f"VIIRS raster not found: {viirs_path}")
        if not counties_path.exists(): 
            raise FileNotFoundError(f"county shapefile not found: {counties_path}")

        spatial_list = []
        mask_list    = []
        gaf_list     = []
        labels       = []
        fips_codes   = []

        with rasterio.open(viirs_path) as src: 
            nodata = src.nodata 
            with fiona.open(counties_path) as shp: 
                count = 0 
                for fips, geom in self._iter_counties(shp, src.crs): 
                    out_image, _ = rasterio.mask.mask(
                        src,
                        [geom],
                        crop=True,
                        filled=False,
                        all_touched=self.all_touched
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

                    spatial, mask = self._recanvas(data, valid_mask, self.canvas_h, self.canvas_w)
                    gaf           = self._gaf_from_mask(data, valid_mask, self.gaf_size)

                    spatial_list.append(spatial)
                    mask_list.append(mask)
                    gaf_list.append(gaf)
                    labels.append(self.label_map[fips])
                    fips_codes.append(fips)

                    if self.debug_png_dir: 
                        self._write_debug_png(fips, spatial, mask, gaf)

                    count += 1 
                    if self.max_counties is not None and count >= self.max_counties: 
                        break 

        if not spatial_list: 
            raise ValueError("no counties processed")

        spatial    = np.stack(spatial_list, axis=0).astype(np.float32) 
        mask       = np.stack(mask_list, axis=0).astype(np.uint8)
        gaf        = np.stack(gaf_list, axis=0).astype(np.float32)
        labels     = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
        fips_codes = np.asarray(fips_codes, dtype="U5") 


        return {
            "spatial": spatial, 
            "mask": mask, 
            "gaf": gaf, 
            "labels": labels,
            "fips": fips_codes 
        }

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
    def _recanvas(data, valid_mask, out_h, out_w): 
        out      = np.zeros((out_h, out_w), dtype=np.float32)
        out_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        ys, xs   = np.nonzero(valid_mask)
        if ys.size == 0: 
            return out, out_mask 

        y0, y1 = ys.min(), ys.max() + 1 
        x0, x1 = xs.min(), xs.max() + 1 

        crop_data = data[y0:y1, x0:x1]
        crop_mask = valid_mask[y0:y1, x0:x1].astype(np.uint8)

        h, w   = crop_data.shape 
        scale  = min(out_h / h, out_w / w) 
        new_h  = max(1, int(round(h * scale)))
        new_w  = max(1, int(round(w * scale)))
        zoom_h = new_h / h 
        zoom_w = new_w / w

        resized_data = zoom(crop_data, (zoom_h, zoom_w), order=1)
        resized_mask = zoom(crop_mask, (zoom_h, zoom_w), order=0)

        out      = np.zeros((out_h, out_w), dtype=np.float32)
        out_mask = np.zeros((out_h, out_w), dtype=np.uint8)

        y_off = (out_h - new_h) // 2 
        x_off = (out_w - new_w) // 2 
        
        out[y_off:y_off + new_h, x_off:x_off + new_w] = resized_data
        out_mask[y_off:y_off + new_h, x_off:x_off + new_w] = resized_mask
 
        return out, out_mask 

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


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--out", default=project_path("data", "datasets", 
                                                      "viirs_nchs_tensor_2023.mat"))
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
