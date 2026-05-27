"""ERA5 hourly maps, interactive.

Anomalies are computed against the monthly climatology of the date's month
(same climatology files as the monthly app, fetched from the GitHub Release
on the ERA5_streamlit repo).
"""

import calendar
import datetime
import numpy as np
import xarray as xr
import streamlit as st

import _common as C


# ─── page setup ───────────────────────────────────────────────────────────────
C.configure_page(
    title="ERA5 · Hourly Maps",
    subtitle="Pick a date and hour (1940 – today). Surface fields and pressure levels.",
    icon="🕒",
)

REPO_URL = "https://github.com/Langosmon/ERA5_hourly_streamlit"


# ─── sidebar controls ────────────────────────────────────────────────────────
choice, domain, code, vname, units, cmap_abs, cmap_anom, plevel = C.variable_picker()

st.sidebar.header("Date & time")
col_d, col_h = st.sidebar.columns([2, 1])
selected_date = col_d.date_input(
    "Date (UTC)",
    value=datetime.date(2023, 10, 25),  # Hurricane Otis landfall
    min_value=datetime.date(1940, 1, 1),
    max_value=datetime.date.today(),
)
selected_hour = col_h.selectbox("Hour", list(range(24)), index=12)

st.sidebar.header("Display")
show_anom  = st.sidebar.toggle("Anomaly (value − monthly climatology)", value=False,
                               help="Anomaly is computed against the climatology "
                                    "mean for the selected month (1980–2010).")
show_sig   = st.sidebar.toggle("Mark statistically significant", value=False,
                               disabled=not show_anom,
                               help="Stipples points where |anomaly| ≥ 1.96·σ. "
                                    "Requires std-dev climatology.")
show_coast = st.sidebar.toggle("Coastlines", value=True)

mask_mode = st.sidebar.radio(
    "Show data on", ("All", "Land", "Ocean"),
    horizontal=True,
    help="Masks the field by ERA5's land-sea mask.",
)


# ─── data loading ────────────────────────────────────────────────────────────
def rda_url(dom: str, code: str, var: str, date: datetime.date) -> str:
    base = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/"
    year, month, day = date.year, date.month, date.day
    if dom == "sfc":
        _, last_day = calendar.monthrange(year, month)
        folder = f"e5.oper.an.sfc/{year}{month:02d}/"
        filename = (f"e5.oper.an.sfc.128_{code}_{var}.ll025sc."
                    f"{year}{month:02d}0100_{year}{month:02d}{last_day}23.nc")
    else:
        extra = "uv" if var in {"u", "v"} else "sc"
        folder = f"e5.oper.an.pl/{year}{month:02d}/"
        filename = (f"e5.oper.an.pl.128_{code}_{var}.ll025{extra}."
                    f"{year}{month:02d}{day:02d}00_{year}{month:02d}{day:02d}23.nc")
    return base + folder + filename


target_dt = datetime.datetime(selected_date.year, selected_date.month,
                              selected_date.day, selected_hour)

try:
    full_chunk = C.load_field_cached(
        rda_url(domain, code, vname, selected_date), vname, plevel,
    )
    da = full_chunk.sel(time=target_dt, method="nearest")
except Exception as e:
    st.error(
        "**Failed to load remote ERA5 data.**  "
        "The RDA server may be temporarily unreachable, or this "
        "date/variable/level combination may not exist."
    )
    st.exception(e)
    st.stop()

da, units = C.apply_unit_conversions(da, vname, units)


# ─── anomaly + significance ──────────────────────────────────────────────────
cmap = cmap_abs
clim_std = None

if show_anom:
    try:
        clim = C.load_climatology(domain, vname, plevel)
        clim_month = clim.sel(month=selected_date.month)
        da = da - clim_month
        cmap = cmap_anom
        units = (units or "") + " anomaly"

        if show_sig:
            if C.climatology_has_std(domain, vname, plevel):
                std_full = C.load_climatology_std(domain, vname, plevel)
                clim_std = std_full.sel(month=selected_date.month)
            else:
                st.info(
                    "**Significance:** the current climatology file contains "
                    "only the mean. Regenerate with std dev (see "
                    "`tools/build_climatology.py` in the ERA5_streamlit repo) "
                    "and reupload to the GitHub Release to enable this overlay."
                )
                show_sig = False
    except FileNotFoundError as e:
        st.warning(f"Climatology unavailable for this field — showing absolute value instead.\n\n{e}")
        show_anom = False


# ─── land / sea mask ─────────────────────────────────────────────────────────
if mask_mode != "All":
    try:
        da = C.apply_lsm_mask(da, mask_mode)
        if clim_std is not None:
            clim_std = C.apply_lsm_mask(clim_std, mask_mode)
    except Exception as e:
        st.warning(f"Couldn't load land-sea mask — showing unmasked field.\n\n{e}")


# ─── region picker + box-select autoscale ────────────────────────────────────
region_bbox, region_name = C.region_picker()

override_default = None
override_label = None
last_box = st.session_state.get("_last_box")

if last_box is not None:
    lat_min, lat_max, lon_min, lon_max = last_box
    if da.longitude.max() > 180 and lon_min < 0:
        lon_min, lon_max = (lon_min + 360) % 360, (lon_max + 360) % 360
    qlo, qhi = C.rescale_to_region(da, lat_min, lat_max, lon_min, lon_max,
                                   symmetric=show_anom)
    override_default = (qlo, qhi)
    override_label = (f"Tuned to box: {lat_min:.1f}–{lat_max:.1f}°N, "
                      f"{lon_min:.1f}–{lon_max:.1f}°E")
elif region_bbox is not None:
    lat_min, lat_max, lon_min, lon_max = region_bbox
    if da.longitude.max() > 180:
        lon_min, lon_max = (lon_min + 360) % 360, (lon_max + 360) % 360
    qlo, qhi = C.rescale_to_region(da, lat_min, lat_max, lon_min, lon_max,
                                   symmetric=show_anom)
    override_default = (qlo, qhi)
    override_label = f"Tuned to 98% of data in: {region_name}"

cmin, cmax = C.colourbar_controls(da, show_anom,
                                  override_default=override_default,
                                  override_label=override_label)


# ─── figure ──────────────────────────────────────────────────────────────────
title = f"{choice} · {selected_date.strftime('%Y-%m-%d')} {selected_hour:02d}:00 UTC"
if plevel is not None: title += f" · {plevel} hPa"
if show_anom: title += " · anomaly"
if mask_mode != "All": title += f" · {mask_mode.lower()} only"

fig = C.build_figure(da, title, units, cmap, cmin, cmax, show_coast, height=580)

if show_anom and show_sig and clim_std is not None:
    C.add_significance_stipple(fig, da, clim_std, z=1.96, stride=8)

event = st.plotly_chart(
    fig, use_container_width=True,
    on_select="rerun", selection_mode=("box",),
    key="main_plot",
    config={"displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d"]},
)

new_box = C.box_selection_to_bounds(event)
if new_box is not None and new_box != last_box:
    st.session_state["_last_box"] = new_box
    st.rerun()

col_t, col_r = st.columns([4, 1])
with col_t:
    st.caption(
        "💡 **Box-select tool** in the Plotly toolbar → draw a region → "
        "colour-bar rescales to the 98% quantile of that box. Useful when "
        "topography or coastal extremes dominate the global range."
    )
with col_r:
    if last_box is not None and st.button("Reset region", use_container_width=True):
        st.session_state["_last_box"] = None
        st.rerun()


with st.expander("Quick picks — notable storms / events", expanded=False):
    st.caption("Common cases worth looking at:")
    st.markdown(
        "- **2023-10-25 06 UTC** — Hurricane Otis at Cat 5 landfall, Acapulco\n"
        "- **2015-10-23 12 UTC** — Hurricane Patricia at peak intensity (185 kt)\n"
        "- **2012-10-29 23 UTC** — Hurricane Sandy NJ landfall\n"
        "- **2017-09-06 12 UTC** — Hurricane Irma in the Caribbean"
    )

C.render_footer(REPO_URL)
