# app.py – ERA5 interactive maps for hourly data with colour-bar sliders
import xarray as xr
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cartopy.feature as cfeature
import datetime
import calendar

st.set_page_config(layout="wide")

# ─────────── constants ────────────────────────────────────────────────────
COMMON_PLEVELS = [975, 850, 700, 500, 250, 100, 50, 10]

# (domain, code, vname, units, cmap)
SURFACE = {
    "Sea-surface temperature": ("sfc", "034", "sstk", "°C",    "thermal"),
    "CAPE"                   : ("sfc", "059", "cape", "J kg⁻¹","viridis"),
    "Surface geopotential"   : ("sfc", "129", "z",    "m² s⁻²","magma"),
    "Surface pressure"       : ("sfc", "134", "sp",   "hPa",   "icefire"),
    "Mean sea-level press."  : ("sfc", "151", "msl",  "hPa",   "icefire"),
    "10-m zonal wind"        : ("sfc", "165", "10u",  "m s⁻¹", "curl"),
    "10-m meridional wind"   : ("sfc", "166", "10v",  "m s⁻¹", "curl_r"),
    "2-m temperature"        : ("sfc", "167", "2t",   "°C",    "thermal"),
}

PRESSURE = {
    "Potential vorticity" : ("pl", "060", "pv", "PVU",     "plasma"),
    "Geopotential"        : ("pl", "129", "z",  "m² s⁻²",  "magma"),
    "Temperature"         : ("pl", "130", "t",  "K",       "thermal"),
    "Zonal wind"          : ("pl", "131", "u",  "m s⁻¹",   "curl"),
    "Meridional wind"     : ("pl", "132", "v",  "m s⁻¹",   "curl_r"),
    "Specific humidity"   : ("pl", "133", "q",  "kg kg⁻¹", "viridis"),
    "Vertical velocity"   : ("pl", "135", "w",  "Pa s⁻¹",  "icefire"),
    "Relative vorticity"  : ("pl", "138", "vo", "s⁻¹",     "plasma"),
    "Divergence"          : ("pl", "155", "d",  "s⁻¹",     "plasma"),
    "Relative humidity"   : ("pl", "157", "r",  "%",       "viridis"),
    "Ozone"               : ("pl", "203", "o3", "kg kg⁻¹", "viridis"),
}

# ─────────── sidebar ─────────────────────────────────────────────────────
st.sidebar.header("Field")
field_type = st.sidebar.radio("Domain", ("Surface", "Pressure level"))

if field_type == "Surface":
    choice = st.sidebar.selectbox("Variable", list(SURFACE))
    domain, code, vname, units, cmap = SURFACE[choice]
    plevel = None
else:
    choice = st.sidebar.selectbox("Variable", list(PRESSURE))
    domain, code, vname, units, cmap = PRESSURE[choice]
    plevel = st.sidebar.selectbox("Pressure level (hPa)", COMMON_PLEVELS)

st.sidebar.markdown("### Date and Time")
selected_date = st.sidebar.date_input(
    "Date",
    value=datetime.date(2022, 1, 1),
    min_value=datetime.date(1940, 1, 1),
    max_value=datetime.date.today(),
)

selected_hour = st.sidebar.selectbox("Hour (UTC)", list(range(24)))
show_coast = st.sidebar.checkbox("Show coastlines", value=True)


# ─────────── helpers ────────────────────────────────────────────────────
def rda_url(dom, code, var, date):
    """Generates the RDA THREDDS URL for hourly ERA5 data."""
    base = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/"
    year, month, day = date.year, date.month, date.day

    if dom == "sfc":
        # Monthly files for surface data
        _, last_day = calendar.monthrange(year, month)
        folder = f"e5.oper.an.sfc/{year}{month:02d}/"
        filename = (f"e5.oper.an.sfc.128_{code}_{var}.ll025sc."
                    f"{year}{month:02d}0100_{year}{month:02d}{last_day}23.nc")
    else:  # dom == "pl"
        # Daily files for pressure level data
        extra = "uv" if var in {"u", "v"} else "sc"
        folder = f"e5.oper.an.pl/{year}{month:02d}/"
        filename = (f"e5.oper.an.pl.128_{code}_{var}.ll025{extra}."
                    f"{year}{month:02d}{day:02d}00_{year}{month:02d}{day:02d}23.nc")
    return base + folder + filename

@st.cache_resource
def open_dataset_from_url(url):
    """Opens and caches a remote xarray dataset."""
    return xr.open_dataset(url)

def find_var(ds, short):
    """Finds the variable name in the dataset, handling different conventions."""
    up = short.upper()
    for k in (up, f"VAR_{up}", up.replace("10", "10M")):
        if k in ds:
            return k
    raise KeyError(f"Variable '{short}' not found in dataset.")

@st.cache_resource
def coastlines_trace(res="110m", gap=10):
    xs, ys = [], []
    feat = cfeature.NaturalEarthFeature("physical", "coastline", res,
                                        edgecolor="black", facecolor="none")
    for geom in feat.geometries():
        for line in getattr(geom, "geoms", [geom]):
            lon, lat = line.coords.xy
            lon = np.mod(lon, 360)
            xs.append(np.nan); ys.append(np.nan)
            for i in range(len(lon)):
                xs.append(lon[i]); ys.append(lat[i])
                if i < len(lon) - 1 and abs(lon[i+1] - lon[i]) > gap:
                    xs.append(np.nan); ys.append(np.nan)
    return go.Scatter(x=xs, y=ys, mode="lines",
                      line=dict(color="black", width=0.8),
                      hoverinfo="skip", showlegend=False)

# ─────────── load hourly field ──────────────────────────────────────────
target_datetime = datetime.datetime(
    selected_date.year, selected_date.month, selected_date.day, selected_hour
)

try:
    url = rda_url(domain, code, vname, selected_date)
    ds = open_dataset_from_url(url)
    da = ds[find_var(ds, vname)].sel(time=target_datetime, method="nearest")
    if plevel is not None:
        da = da.sel(level=plevel)

    # Unit conversions
    if vname in {"sstk", "2t", "t"}: da, units = da - 273.15, "°C"
    if vname in {"sp", "msl"}:       da, units = da / 100.0, "hPa"

except Exception as e:
    st.error(f"""
    **Failed to load data.** This could be due to a few reasons:
    - The remote data server might be temporarily down.
    - The requested date/variable combination might not be available.
    - There might be a network issue.

    Please try a different date or variable.

    **Error details:** `{e}`
    """)
    st.stop()


# ─────────── colour-bar controls ──────────────────────────────────────────
data_min = float(np.nanmin(da))
data_max = float(np.nanmax(da))
default_min, default_max = data_min, data_max

st.sidebar.markdown("### Colour-bar limits")
step = (default_max - default_min) / 50 or 1e-6  # avoid step=0
cmin = st.sidebar.slider("Min", data_min, data_max, value=default_min, step=step, format="%.4g")
cmax = st.sidebar.slider("Max", data_min, data_max, value=default_max, step=step, format="%.4g")

# Auto-scale button (98 % central quantile)
if st.sidebar.button("Auto-scale (98 % of data)"):
    qmin, qmax = np.nanquantile(da, [0.01, 0.99])
    cmin, cmax = float(qmin), float(qmax)

if cmin >= cmax:
    st.sidebar.error("Min must be less than Max")
    st.stop()

# ─────────── plot ──────────────────────────────────────────────────────
title_date = f"{selected_date.strftime('%Y-%m-%d')} {selected_hour:02d}:00 UTC"
title = f"{choice} • {title_date}" + (f" • {plevel} hPa" if plevel else "")

fig = px.imshow(
    da,
    origin="lower",
    aspect="auto",
    color_continuous_scale=cmap,
    labels=dict(color=units),
    title=title,
)
fig.update_coloraxes(cmin=cmin, cmax=cmax)
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), uirevision="keep")

if show_coast:
    fig.add_trace(coastlines_trace())

st.plotly_chart(fig, use_container_width=True)

st.caption("Plot created reading ERA5 from NCAR's RDA. Code by [Jose A. Ocegueda Sanchez](https://Langosmon.github.io).")






