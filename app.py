"""ERA5 hourly maps, interactive.

Anomalies are computed against the monthly climatology of the date's month
(same climatology files as the monthly app, fetched from the GitHub Release
on the ERA5_streamlit repo).

Unit contract: fields and climatology are differenced in ERA5 NATIVE units;
apply_unit_conversions() runs AFTER the anomaly step (see _common docstring).

Note on statistics: this app deliberately has NO significance overlay. The
available σ is the year-to-year spread of MONTHLY means — the wrong noise
floor for an hourly snapshot (hourly synoptic + diurnal variability is far
larger), so a stipple here would mark nearly everything "significant".
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
    subtitle="Pick a date and hour (1940 – near-present). Surface fields and pressure levels.",
    icon="🕒",
)

REPO_URL = "https://github.com/Langosmon/ERA5_hourly_streamlit"

# ERA5 has ~5-day latency; leave headroom so the default range always exists.
LATEST_DAY = datetime.date.today() - datetime.timedelta(days=6)


# ─── sidebar controls ────────────────────────────────────────────────────────
choice, domain, code, vname, units, cmap_abs, cmap_anom, plevel = C.variable_picker()

st.sidebar.header("Date & time")
col_d, col_h = st.sidebar.columns([2, 1])
selected_date = col_d.date_input(
    "Date (UTC)",
    value=datetime.date(2023, 10, 25),  # Hurricane Otis landfall
    min_value=datetime.date(1940, 1, 1),
    max_value=LATEST_DAY,
)
selected_hour = col_h.selectbox("Hour", list(range(24)), index=6)  # Otis landfall ≈ 06:25 UTC
st.sidebar.caption("ERA5 publishes with ~5 days of latency.")

st.sidebar.header("Display")
animate = st.sidebar.toggle("▶ Day animation (24 h)", value=False,
                            help="Loads all 24 hours of the selected day at "
                                 "0.5° resolution and plays them in your "
                                 "browser — play/pause and an hour slider "
                                 "appear under the map. First load of a day "
                                 "transfers ~25 MB from RDA, then it's cached.")
show_anom  = st.sidebar.toggle("Anomaly (value − monthly climatology)", value=False,
                               help="Departure from the 1980–2010 mean of the "
                                    "selected month. Because the reference is a "
                                    "monthly mean, hourly anomalies still contain "
                                    "the diurnal cycle — largest for 2-m "
                                    "temperature over land (±10 K or more).")
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
    if animate:
        # All 24 hours of the day at 0.5° (stride 2) — OPeNDAP subsets
        # server-side, so this transfers ~25 MB instead of ~100 MB.
        da = C.load_field_cached(
            rda_url(domain, code, vname, selected_date), vname, plevel,
            day_sel=selected_date.isoformat(), stride=2,
        )
    else:
        # time_sel slices server-side BEFORE download: one ~4 MB hour instead
        # of a whole month of hourly fields (~3 GB for surface files).
        da = C.load_field_cached(
            rda_url(domain, code, vname, selected_date), vname, plevel,
            time_sel=target_dt.isoformat(),
        )
except Exception as e:
    st.error(
        "**Failed to load remote ERA5 data.**  "
        "The RDA server may be temporarily unreachable, or this "
        "date/variable/level combination may not exist."
    )
    with st.expander("Technical details"):
        st.exception(e)
    st.stop()


# ─── anomaly (native units) ──────────────────────────────────────────────────
cmap = cmap_abs

if show_anom:
    try:
        clim = C.load_climatology(domain, vname, plevel)
        clim_month = clim.sel(month=selected_date.month)
        da = da - clim_month          # native − native: units match
        cmap = cmap_anom
    except Exception as e:
        st.warning(f"Climatology unavailable for this field — showing absolute value instead.\n\n{e}")
        show_anom = False
        cmap = cmap_abs

# Convert for display AFTER the subtraction (see module docstring).
da, units = C.apply_unit_conversions(da, vname, units, anomaly=show_anom)
if show_anom:
    units = (units or "") + " anomaly"


# ─── land / sea mask ─────────────────────────────────────────────────────────
if mask_mode != "All":
    try:
        da = C.apply_lsm_mask(da, mask_mode)
        if bool(np.all(np.isnan(da.values))):
            st.info(f"**Nothing to show:** {choice} has no data over "
                    f"{mask_mode.lower()} (e.g. SST is ocean-only). "
                    "Switch \"Show data on\" back to All or the other side.")
    except Exception as e:
        st.warning(f"Couldn't load land-sea mask — showing unmasked field.\n\n{e}")


# ─── region focus: preset picker + box-select ────────────────────────────────
region_bbox, region_name = C.region_picker()

# Read a freshly drawn box from the widget state BEFORE building the figure —
# one rerun stores and applies it (no second st.rerun round-trip).
state_box = C.box_selection_to_bounds(st.session_state.get("main_plot"))
if state_box is not None and state_box != st.session_state.get("_dismissed_box"):
    st.session_state["_last_box"] = state_box
last_box = st.session_state.get("_last_box")

override_default = None
override_label = None

if last_box is not None:
    lat_min, lat_max, lon_min, lon_max = last_box
    qlo, qhi = C.rescale_to_region(da, lat_min, lat_max, lon_min, lon_max,
                                   symmetric=show_anom)
    override_default = (qlo, qhi)
    override_label = (f"Tuned to box: {lat_min:.1f}–{lat_max:.1f}°N, "
                      f"{lon_min:.1f}–{lon_max:.1f}°E")
elif region_bbox is not None:
    lat_min, lat_max, lon_min, lon_max = region_bbox
    qlo, qhi = C.rescale_to_region(da, lat_min, lat_max, lon_min, lon_max,
                                   symmetric=show_anom)
    override_default = (qlo, qhi)
    override_label = f"Tuned to 98% of data in: {region_name}"

cmin, cmax = C.colourbar_controls(da, show_anom,
                                  override_default=override_default,
                                  override_label=override_label)


# ─── figure ──────────────────────────────────────────────────────────────────
title = f"{choice} · {selected_date.strftime('%Y-%m-%d')}"
if not animate: title += f" {selected_hour:02d}:00 UTC"
if plevel is not None: title += f" · {plevel} hPa"
if show_anom: title += " · anomaly"
if mask_mode != "All": title += f" · {mask_mode.lower()} only"

if animate:
    fig = C.build_animation_figure(da, title + " · 24 h", units, cmap,
                                   cmin, cmax, show_coast, height=580)
    st.plotly_chart(
        fig, use_container_width=True, key="anim_plot",
        config={"displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.caption(
        "▶ Press play (bottom left) or drag the hour slider. Frames are "
        "0.5° and colour-mapped server-side, so hover values are off in "
        "animation mode — switch the toggle off to inspect exact values. "
        "Region presets still tune the colour-bar."
    )
else:
    fig = C.build_figure(da, title, units, cmap, cmin, cmax, show_coast, height=580)

    st.plotly_chart(
        fig, use_container_width=True,
        on_select="rerun", selection_mode=("box",),
        key="main_plot",
        config={"displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d"]},
    )

if show_anom:
    st.caption(
        "⚠️ Anomaly = value minus the **monthly-mean** climatology, so the "
        "diurnal cycle is still in the signal (strongest for 2-m temperature "
        "over land). Compare the same hour across days/years for a cleaner read."
    )

if not animate:
    col_t, col_r = st.columns([4, 1])
    with col_t:
        st.caption(
            "💡 **Box-select tool** in the Plotly toolbar → draw a region → "
            "colour-bar rescales to the 98% quantile of that box. Useful when "
            "topography or coastal extremes dominate the global range."
        )
    with col_r:
        if last_box is not None and st.button("Reset region", use_container_width=True):
            st.session_state["_dismissed_box"] = last_box
            st.session_state["_last_box"] = None
            st.rerun()
elif last_box is not None and st.button("Reset region", use_container_width=False):
    st.session_state["_dismissed_box"] = last_box
    st.session_state["_last_box"] = None
    st.rerun()


with st.expander("Quick picks — notable storms / events", expanded=False):
    st.caption("Common cases worth looking at:")
    st.markdown(
        "- **2023-10-25 06 UTC** — Hurricane Otis at Cat 5 landfall, Acapulco\n"
        "- **2015-10-23 12 UTC** — Hurricane Patricia at peak intensity (185 kt)\n"
        "- **2012-10-29 23 UTC** — Post-tropical Sandy NJ landfall\n"
        "- **2017-09-06 12 UTC** — Hurricane Irma in the Caribbean"
    )

C.render_footer(REPO_URL)
