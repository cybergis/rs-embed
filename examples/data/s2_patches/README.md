# Sample Sentinel-2 patches for `bring_your_own_data.ipynb`

Eight small GeoTIFFs standing in for "imagery you already have". Each file is a 192 x 192 px, 12-band Sentinel-2 L2A patch (`COPERNICUS/S2_SR_HARMONIZED`), median composite over 2022-06-01 .. 2022-09-01 with cloudy_pct 30, sampled at 10 m on the EPSG:3857 grid, stored as uint16 surface-reflectance DN (0..10000) in the canonical band order `B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12` (also written as band descriptions and file tags).

| Prefix      | Centre (lon, lat) | Note                                  |
| ----------- | ----------------- | ------------------------------------- |
| `shanghai_` | 121.5 E, 31.2 N   | urban core, Huangpu river             |
| `zhejiang_` | 120.5 E, 30.2 N   | farmland and waterways, Zhejiang      |

`q0..q3` are the four quadrants of one 4 km x 4 km fetch (top-left, top-right, bottom-left, bottom-right). They were cut from the `input_chw` arrays saved by `export_batch(..., save_inputs=True)`, i.e. exactly what an rs-embed provider fetch returns, so the values are in raw provider units.
