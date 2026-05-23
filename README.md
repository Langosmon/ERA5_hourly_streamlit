# ERA5_hourly_streamlit

Interactive Streamlit app for ERA5 hourly maps, reading data from NCAR's RDA
via OPeNDAP. Companion to
[ERA5_streamlit](https://github.com/Langosmon/ERA5_streamlit) (monthly).

Live: https://era5hourlyapp-alfredocegueda.streamlit.app/

## Features

- 8 surface fields + 11 pressure-level fields, 8 standard pressure levels
- Hourly resolution, **1940 – today**
- Quick-pick presets for notable storms (Otis, Patricia, Sandy, Irma)
- **Anomaly** toggle: departure from the monthly climatology of the selected
  month (climatology hosted on `ERA5_streamlit` GitHub Releases — fetched on
  demand)
- **Statistical significance** overlay (requires std-dev climatology)
- Coastline overlay (no cartopy dependency)

## Architecture

```
app.py            Streamlit app
_common.py        Shared helpers (mirrored from ERA5_streamlit)
coastlines.json   Precomputed Natural Earth 110m coastlines
.streamlit/
  config.toml     Theme tokens matching langosmon.github.io
```

Climatology files are pulled from the GitHub Release on `ERA5_streamlit`
(tag `climatology-v1`). The configuration lives in `_common.py` — if you bump
the tag in that repo, bump it here too (keep `_common.py` in sync between
the two repos).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## License & Citation

Apache License 2.0. If you use this in academic work, please cite:

> Jose A. Ocegueda Sanchez. *ERA5 hourly Streamlit.* https://github.com/Langosmon/ERA5_hourly_streamlit

See the `NOTICE` file for additional attribution guidelines.

## Contact

- jocegue@purdue.edu
- [LinkedIn](https://www.linkedin.com/in/josé-alfredo-ocegueda-sanchez-a3598b122/)
- Personal site: https://langosmon.github.io
