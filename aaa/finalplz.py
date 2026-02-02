# ============================================================
# SSA FULL FINAL — "All Features" Streamlit + Skyfield + Plotly
# ============================================================
# ✅ Fixes included:
#   - Skyfield version differences: itrs_xyz method/attr missing
#   - Distance / Vector object conversion robustness
#   - Streamlit rerun API compatibility (st.rerun / st.experimental_rerun)
#   - LIVE ON/OFF to stop constant refresh during mouse interaction
#   - Plotly uirevision keep to preserve camera/zoom while updating
#   - Sidebar visibility protection (CSS)
#
# ✅ Features included (no reduction):
#   - 3D Earth sphere + lat/lon grid
#   - Global constellation distribution: GPS / STARLINK / GALILEO / GLONASS / BEIDOU / ISS
#   - Ground station marker
#   - Visibility filter by min elevation + VISIBLE ONLY mode
#   - Line-of-sight lines to visible sats (option: only when tracking)
#   - Trails (window minutes, step seconds, top N per group)
#   - Track cam + target halo + dim other sats
#   - Best link recommendation (elev + range)
#   - Pass prediction table (rise/culm/set) + timeline (KST)
#   - Next-pass ground track overlay on Earth surface
# ============================================================

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Skyfield
from skyfield.api import load, wgs84
from skyfield.framelib import itrs


# =========================
# USER SETTINGS (editable)
# =========================
GROUND_LAT = 37.3946
GROUND_LON = 127.1117
ELEV_M = 50

EARTH_RADIUS_KM = 6371.0
KST = timezone(timedelta(hours=9))

# CelesTrak TLE sources
TLE_SOURCES = {
    "GPS": "https://celestrak.org/NORAD/elements/gps-ops.txt",
    "STARLINK": "https://celestrak.org/NORAD/elements/gp.php?FORMAT=tle&GROUP=starlink",
    "GALILEO": "https://celestrak.org/NORAD/elements/galileo.txt",
    "GLONASS": "https://celestrak.org/NORAD/elements/glo-ops.txt",
    "BEIDOU": "https://celestrak.org/NORAD/elements/beidou.txt",
    "ISS/Stations": "https://celestrak.org/NORAD/elements/stations.txt",
}


# =========================
# USSF-ish THEME
# =========================
HUD_BG = "#060B16"
HUD_PANEL = "rgba(10,18,40,0.65)"
HUD_GRID = "rgba(80,180,255,0.12)"

HUD_CYAN = "#46D5FF"      # GPS
HUD_GREEN = "#58FFB1"     # Starlink
HUD_PURPLE = "#B48CFF"    # Galileo
HUD_WHITE = "rgba(255,255,255,0.85)"  # GLONASS
HUD_PINK = "rgba(255,160,220,0.85)"   # BeiDou
HUD_GOLD = "rgba(255,230,150,0.90)"   # ISS
HUD_WARN = "#FFB86B"
HUD_RED = "#FF5C7A"
HUD_TEXT = "#CFE8FF"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"

GROUP_COLOR = {
    "GPS": HUD_CYAN,
    "STARLINK": HUD_GREEN,
    "GALILEO": HUD_PURPLE,
    "GLONASS": HUD_WHITE,
    "BEIDOU": HUD_PINK,
    "ISS/Stations": HUD_GOLD,
}


# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="SSA FULL FINAL", layout="wide")

# ---- CSS: enforce sidebar visibility + USSF UI
st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {HUD_BG};
}}
/* Sidebar force-visible (prevents 'disappearing sidebar' due to custom CSS) */
[data-testid="stSidebar"] {{
  display: block !important;
  visibility: visible !important;
  width: 21rem !important;
  min-width: 21rem !important;
  background: linear-gradient(180deg, rgba(10,18,40,0.92), rgba(6,11,22,0.92)) !important;
  border-right: 1px solid rgba(70,213,255,0.15) !important;
}}
/* Text colors */
h1, h2, h3, p, label, span, div {{
    color: {HUD_TEXT} !important;
}}
/* Cards */
.hud-card {{
    background: {HUD_PANEL};
    border: 1px solid rgba(70,213,255,0.16);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 0 24px rgba(70,213,255,0.08);
}}
.hud-status {{
    font-family: {FONT_MONO};
    letter-spacing: 0.6px;
    color: {HUD_CYAN} !important;
}}
.hud-mono {{ font-family: {FONT_MONO}; }}
.small {{
    font-size: 12px;
    opacity: 0.85;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🛰️ SSA ORBITAL TRACKING CONSOLE — FULL FINAL BUILD")


# =========================
# Robust Skyfield helpers
# =========================
def xyz_km_from_skyfield(obj) -> Tuple[float, float, float]:
    """
    Robust extraction of (x,y,z) km across Skyfield versions/types.
    Preference: ITRS/ECEF (Earth-fixed), fallback to any cartesian available.
    Handles:
      - obj.itrs_xyz being method or attribute
      - obj.frame_xyz(itrs) (safe across many versions)
      - obj.position.km
      - obj.xyz.km (rare)
    """
    # 1) itrs_xyz attr/method (some versions)
    itrs_attr = getattr(obj, "itrs_xyz", None)
    if itrs_attr is not None:
        try:
            v = itrs_attr() if callable(itrs_attr) else itrs_attr
            if hasattr(v, "km"):
                x, y, z = v.km
                return float(x), float(y), float(z)
        except Exception:
            pass

    # 2) frame_xyz(itrs) is broadly supported
    frame_xyz = getattr(obj, "frame_xyz", None)
    if callable(frame_xyz):
        try:
            v = obj.frame_xyz(itrs)
            if hasattr(v, "km"):
                x, y, z = v.km
                return float(x), float(y), float(z)
            if hasattr(v, "position") and hasattr(v.position, "km"):
                x, y, z = v.position.km
                return float(x), float(y), float(z)
        except Exception:
            pass

    # 3) fallback: position.km
    if hasattr(obj, "position") and hasattr(obj.position, "km"):
        x, y, z = obj.position.km
        return float(x), float(y), float(z)

    # 4) fallback: xyz.km
    xyz = getattr(obj, "xyz", None)
    if xyz is not None and hasattr(xyz, "km"):
        x, y, z = xyz.km
        return float(x), float(y), float(z)

    raise AttributeError("Cannot extract xyz km from Skyfield object (version mismatch).")


def sat_altaz(sat, obs, t) -> Tuple[float, float, float]:
    """Return (elev_deg, az_deg, range_km)"""
    diff = sat - obs
    alt, az, dist = diff.at(t).altaz()
    return float(alt.degrees), float(az.degrees), float(dist.km)


# =========================
# Earth mesh + grid
# =========================
def earth_mesh(r=EARTH_RADIUS_KM, n_lat=48, n_lon=96):
    lats = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lons = np.linspace(-np.pi, np.pi, n_lon)
    latg, long = np.meshgrid(lats, lons, indexing="ij")
    x = r * np.cos(latg) * np.cos(long)
    y = r * np.cos(latg) * np.sin(long)
    z = r * np.sin(latg)
    return x, y, z


def add_latlon_grid(fig: go.Figure):
    # Longitude lines
    for lon_deg in range(-150, 181, 30):
        lon = math.radians(lon_deg)
        lats = np.linspace(-math.pi / 2, math.pi / 2, 240)
        x = EARTH_RADIUS_KM * np.cos(lats) * np.cos(lon)
        y = EARTH_RADIUS_KM * np.cos(lats) * np.sin(lon)
        z = EARTH_RADIUS_KM * np.sin(lats)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=HUD_GRID, width=2),
            hoverinfo="skip",
            showlegend=False
        ))

    # Latitude lines
    for lat_deg in range(-60, 61, 30):
        lat = math.radians(lat_deg)
        lons = np.linspace(-math.pi, math.pi, 360)
        x = EARTH_RADIUS_KM * np.cos(lat) * np.cos(lons)
        y = EARTH_RADIUS_KM * np.cos(lat) * np.sin(lons)
        z = np.full_like(lons, EARTH_RADIUS_KM * np.sin(lat))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=HUD_GRID, width=2),
            hoverinfo="skip",
            showlegend=False
        ))


def compute_track_camera(sat_xyz: Tuple[float, float, float], distance_factor=2.4):
    x, y, z = sat_xyz
    norm = math.sqrt(x * x + y * y + z * z) + 1e-9
    ux, uy, uz = x / norm, y / norm, z / norm
    return dict(eye=dict(x=ux * distance_factor, y=uy * distance_factor, z=uz * distance_factor))


# =========================
# Loading satellites (cached)
# =========================
@st.cache_resource(show_spinner=False)
def load_all_sats():
    ts = load.timescale()
    groups = {k: load.tle_file(url, reload=True) for k, url in TLE_SOURCES.items()}
    return ts, groups


def make_observer():
    return wgs84.latlon(GROUND_LAT, GROUND_LON, elevation_m=ELEV_M)


# =========================
# Trails
# =========================
def build_trails_xyz(sats, ts_obj, dt_list, limit):
    """
    dt_list: list of python datetimes (UTC)
    returns list: (sat_name, Nx3 xyz km)
    """
    t_list = ts_obj.from_datetimes(dt_list)
    trails = []
    for sat in sats[:limit]:
        pts = []
        for tt in t_list:
            sat_at = sat.at(tt)
            xyz = xyz_km_from_skyfield(sat_at)
            pts.append(xyz)
        trails.append((sat.name, np.array(pts)))
    return trails


# =========================
# Pass prediction + timeline
# =========================
def next_passes_table(ts_obj, sat, obs, min_elev_deg, hours=24, max_passes=8):
    t0 = ts_obj.now()
    t1 = ts_obj.from_datetime(t0.utc_datetime() + timedelta(hours=hours))

    t, events = sat.find_events(obs, t0, t1, altitude_degrees=min_elev_deg)

    rows = []
    i = 0
    while i < len(events) - 2 and len(rows) < max_passes:
        if events[i] == 0 and events[i+1] == 1 and events[i+2] == 2:
            tr, tc, ts_ = t[i], t[i+1], t[i+2]
            alt_c, az_c, _ = (sat - obs).at(tc).altaz()

            tr_kst = tr.utc_datetime().replace(tzinfo=timezone.utc).astimezone(KST)
            tc_kst = tc.utc_datetime().replace(tzinfo=timezone.utc).astimezone(KST)
            ts_kst = ts_.utc_datetime().replace(tzinfo=timezone.utc).astimezone(KST)

            rows.append({
                "rise(KST)": tr_kst.strftime("%m-%d %H:%M:%S"),
                "culm(KST)": tc_kst.strftime("%m-%d %H:%M:%S"),
                "set(KST)":  ts_kst.strftime("%m-%d %H:%M:%S"),
                "max_elev(°)": round(float(alt_c.degrees), 1),
                "az_at_culm(°)": round(float(az_c.degrees), 1),
            })
            i += 3
        else:
            i += 1

    return pd.DataFrame(rows)


def pass_timeline_figure(df, title):
    if df.empty:
        return None

    year = datetime.now(KST).year

    def parse_kst(mmdd_hms: str):
        # row format: "MM-DD HH:MM:SS"
        return datetime.strptime(f"{year}-{mmdd_hms}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)

    times, labels, yvals = [], [], []
    for idx, row in df.iterrows():
        tr = parse_kst(row["rise(KST)"])
        tc = parse_kst(row["culm(KST)"])
        ts_ = parse_kst(row["set(KST)"])

        times += [tr, tc, ts_]
        labels += [
            f"PASS {idx+1} RISE",
            f"PASS {idx+1} CULM (max {row['max_elev(°)']}°)",
            f"PASS {idx+1} SET"
        ]
        yvals += [idx+1, idx+1, idx+1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=yvals, mode="markers+lines",
        text=labels,
        hovertemplate="%{text}<br>%{x}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=HUD_BG,
        plot_bgcolor=HUD_BG,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=title, x=0.02, xanchor="left", font=dict(color=HUD_TEXT, family=FONT_MONO)),
        xaxis=dict(title="Time (KST)", gridcolor=HUD_GRID, color=HUD_TEXT),
        yaxis=dict(title="Pass #", gridcolor=HUD_GRID, color=HUD_TEXT, dtick=1),
        height=280
    )
    return fig


# =========================
# Next-pass ground track overlay
# =========================
def next_pass_ground_track_xyz(ts_obj, sat, obs, min_elev_deg, lookahead_hours=24, samples=90):
    t0 = ts_obj.now()
    t1 = ts_obj.from_datetime(t0.utc_datetime() + timedelta(hours=lookahead_hours))
    t, events = sat.find_events(obs, t0, t1, altitude_degrees=min_elev_deg)

    idx = None
    for i in range(len(events) - 2):
        if events[i] == 0 and events[i+1] == 1 and events[i+2] == 2:
            idx = i
            break
    if idx is None:
        return None

    tr = t[idx]
    ts_ = t[idx+2]
    t_samples = ts_obj.linspace(tr, ts_, samples)

    xs, ys, zs = [], [], []
    for tt in t_samples:
        sp = sat.at(tt).subpoint()
        lat = math.radians(sp.latitude.degrees)
        lon = math.radians(sp.longitude.degrees)
        xs.append(EARTH_RADIUS_KM * math.cos(lat) * math.cos(lon))
        ys.append(EARTH_RADIUS_KM * math.cos(lat) * math.sin(lon))
        zs.append(EARTH_RADIUS_KM * math.sin(lat))

    return np.array(xs), np.array(ys), np.array(zs)


# =========================
# Best link candidate
# =========================
def best_link_candidate(visible_rows):
    if not visible_rows:
        return None

    elevs = np.array([r["elev"] for r in visible_rows], dtype=float)
    dists = np.array([r["dist_km"] for r in visible_rows], dtype=float)

    elev_min, elev_max = float(elevs.min()), float(elevs.max())
    dist_min, dist_max = float(dists.min()), float(dists.max())

    def norm(x, a, b):
        if b - a < 1e-9:
            return 0.0
        return (x - a) / (b - a)

    best = None
    best_score = -1e9
    for r in visible_rows:
        elev_score = norm(r["elev"], elev_min, elev_max)
        dist_score = 1.0 - norm(r["dist_km"], dist_min, dist_max)
        score = 0.70 * elev_score + 0.30 * dist_score
        if score > best_score:
            best_score = score
            best = dict(r)
            best["score"] = float(score)
    return best


# =========================
# Figure builder
# =========================
def make_fig(
    ground_xyz,
    sat_points_by_group,
    hover_by_group,
    los_lines,
    trails_by_group,
    highlight_target=None,      # (x,y,z,label)
    show_earth=True,
    camera=None,
    title="ORBITAL TRACKING CONSOLE — GLOBAL DISTRIBUTION",
    dim_others=False,
    ground_track_xyz=None,      # (xarr,yarr,zarr)
):
    fig = go.Figure()

    # Earth
    if show_earth:
        ex, ey, ez = earth_mesh()
        fig.add_trace(go.Surface(
            x=ex, y=ey, z=ez,
            opacity=0.18,
            showscale=False,
            hoverinfo="skip",
            colorscale=[[0, "#0A1024"], [1, "#0F1B3E"]],
            lighting=dict(ambient=0.65, diffuse=0.25, specular=0.15, roughness=0.90),
        ))
        add_latlon_grid(fig)

    # Ground marker
    gx, gy, gz = ground_xyz
    fig.add_trace(go.Scatter3d(
        x=[gx], y=[gy], z=[gz],
        mode="markers+text",
        text=["GROUND STATION"],
        textposition="top center",
        marker=dict(size=7, color=HUD_WARN, line=dict(width=2, color="rgba(255,220,170,0.95)")),
        name="Ground",
        hovertemplate=f"GROUND STATION<br>lat={GROUND_LAT}, lon={GROUND_LON}<extra></extra>"
    ))

    # LoS lines (visible)
    if los_lines:
        lx, ly, lz = [], [], []
        for (_, p1, p2, _) in los_lines:
            lx += [p1[0], p2[0], None]
            ly += [p1[1], p2[1], None]
            lz += [p1[2], p2[2], None]
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="lines",
            line=dict(color="rgba(255,184,107,0.45)", width=2),
            name="LoS (visible)",
            hoverinfo="skip"
        ))

    # Trails
    for group, trails in trails_by_group.items():
        col = GROUP_COLOR.get(group, "rgba(200,200,255,0.25)")
        for (_, pts) in trails:
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines",
                line=dict(color=col, width=2),
                opacity=0.22,
                showlegend=False,
                hoverinfo="skip"
            ))

    # Satellites
    for group, pts in sat_points_by_group.items():
        if pts.size == 0:
            continue

        col = GROUP_COLOR.get(group, "rgba(200,200,255,0.85)")
        opacity = 0.85
        if dim_others and highlight_target is not None:
            opacity = 0.22

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=4 if group == "GPS" else 3, color=col, opacity=opacity),
            name=group,
            text=hover_by_group.get(group, []),
            hovertemplate="%{text}<extra></extra>"
        ))

    # Next pass ground track overlay (Earth surface)
    if ground_track_xyz is not None:
        gx2, gy2, gz2 = ground_track_xyz
        fig.add_trace(go.Scatter3d(
            x=gx2, y=gy2, z=gz2,
            mode="lines",
            line=dict(color="rgba(255,184,107,0.85)", width=6),
            name="NEXT PASS GROUND TRACK",
            hoverinfo="skip"
        ))

    # Highlight target (halo + marker)
    if highlight_target is not None:
        tx, ty, tz, tlabel = highlight_target
        r = 250.0
        theta = np.linspace(0, 2 * np.pi, 90)
        hx = tx + r * np.cos(theta)
        hy = ty + r * np.sin(theta)
        hz = np.full_like(theta, tz)

        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz,
            mode="lines",
            line=dict(color="rgba(255,184,107,0.55)", width=4),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter3d(
            x=[tx], y=[ty], z=[tz],
            mode="markers+text",
            text=[tlabel],
            textposition="top center",
            marker=dict(size=8, color=HUD_WARN, opacity=0.95, line=dict(width=2, color="rgba(255,220,170,0.9)")),
            name="TARGET",
            hovertemplate=f"{tlabel}<extra></extra>"
        ))

    # Layout
    view_r = EARTH_RADIUS_KM * 2.2
    fig.update_layout(
        paper_bgcolor=HUD_BG,
        plot_bgcolor=HUD_BG,
        margin=dict(l=0, r=0, t=46, b=0),
        title=dict(text=title, x=0.02, xanchor="left",
                   font=dict(color=HUD_TEXT, size=16, family=FONT_MONO)),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.01,
            xanchor="left", x=0.02,
            font=dict(color=HUD_TEXT, family=FONT_MONO)
        ),
        scene=dict(
            bgcolor=HUD_BG,
            xaxis=dict(range=[-view_r, view_r], gridcolor=HUD_GRID, zerolinecolor=HUD_GRID, color=HUD_TEXT, title="X km"),
            yaxis=dict(range=[-view_r, view_r], gridcolor=HUD_GRID, zerolinecolor=HUD_GRID, color=HUD_TEXT, title="Y km"),
            zaxis=dict(range=[-view_r, view_r], gridcolor=HUD_GRID, zerolinecolor=HUD_GRID, color=HUD_TEXT, title="Z km"),
            aspectmode="data",
        ),
        # ✅ Keep camera/zoom state across reruns
        uirevision="keep",
    )
    if camera is not None:
        fig.update_layout(scene_camera=camera)

    return fig


# =========================
# Load data
# =========================
ts, groups = load_all_sats()
obs = make_observer()

# Build target lookup (for tracking select)
all_targets = ["(NONE)"]
sat_lookup = {}
for g, sats in groups.items():
    for s in sats:
        key = f"{g} | {s.name}"
        all_targets.append(key)
        sat_lookup[key] = s


# =========================
# Sidebar (FULL CONTROL)
# =========================
with st.sidebar:
    st.markdown("### LAYERS (Toggle)")
    enabled = {}
    for k in groups.keys():
        enabled[k] = st.checkbox(k, value=(k in ("GPS", "STARLINK")))

    st.divider()
    st.markdown("### LIVE CONTROL")

    if "live" not in st.session_state:
        st.session_state.live = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ START LIVE"):
            st.session_state.live = True
    with c2:
        if st.button("■ STOP LIVE"):
            st.session_state.live = False

    st.caption(f"LIVE MODE: {'ON' if st.session_state.live else 'OFF'}")

    st.divider()
    st.markdown("### GLOBAL / FILTER")
    show_only_visible = st.checkbox("VISIBLE ONLY (3D points)", value=False)
    min_elev = st.slider("MIN ELEV (deg)", 0.0, 60.0, 10.0, 1.0)
    update_sec = st.slider("UPDATE (sec)", 1, 10, 2, 1)

    st.divider()
    st.markdown("### POINT LIMIT (PER GROUP)")
    max_per_group = {}
    for k in groups.keys():
        if k == "STARLINK":
            max_per_group[k] = st.slider("STARLINK MAX", 50, 3000, 900, 50)
        elif k == "GPS":
            max_per_group[k] = st.slider("GPS MAX", 10, 128, 64, 1)
        else:
            max_per_group[k] = st.slider(f"{k} MAX", 10, 800, 200, 10)

    st.divider()
    st.markdown("### FEATURES")
    show_earth = st.checkbox("RENDER EARTH", value=True)
    show_los = st.checkbox("LINE OF SIGHT (visible sats)", value=True)
    los_only_when_tracking = st.checkbox("LoS only when tracking", value=False)

    st.divider()
    st.markdown("### TRAILS")
    show_trails = st.checkbox("SHOW TRAILS", value=True)
    trail_minutes = st.slider("TRAIL WINDOW (minutes)", 1, 30, 10, 1)
    trail_step_sec = st.slider("TRAIL STEP (sec)", 10, 180, 60, 10)
    trail_top_n = st.slider("TRAIL TOP N (per group)", 1, 40, 10, 1)

    st.divider()
    st.markdown("### TRACK CAM")
    track_target = st.selectbox("TRACK TARGET", all_targets, index=0)
    cam_distance = st.slider("CAM DIST (factor)", 1.6, 3.5, 2.4, 0.1)
    dim_others_on_lock = st.checkbox("DIM OTHERS WHEN LOCKED", value=True)

    st.divider()
    st.markdown("### PASS PREDICTION")
    pass_hours = st.slider("LOOKAHEAD (hours)", 1, 48, 24, 1)
    pass_count = st.slider("MAX PASSES", 1, 20, 8, 1)

    st.divider()
    st.markdown("### GROUND STATION")
    st.write(f"LAT {GROUND_LAT}")
    st.write(f"LON {GROUND_LON}")
    st.write(f"ELEV {ELEV_M} m")


# =========================
# Compute one frame
# =========================
t = ts.now()
utc_str = t.utc_strftime("%Y-%m-%d %H:%M:%S")
kst_str = datetime.now(timezone.utc).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")

# Ground XYZ (Earth-fixed)
ground_xyz = xyz_km_from_skyfield(obs.at(t))

# Tracking sat
tracking_sat = None
tracking_key = None
if track_target != "(NONE)":
    tracking_key = track_target
    tracking_sat = sat_lookup.get(tracking_key)

sat_points_by_group: Dict[str, np.ndarray] = {}
hover_by_group: Dict[str, List[str]] = {}
visible_rows = []
los_lines = []
top_sats_for_trails: Dict[str, List] = {}

for g, sats in groups.items():
    sat_points_by_group[g] = np.zeros((0, 3))
    hover_by_group[g] = []

    if not enabled.get(g, False):
        continue

    # rank by elevation, then take max_per_group
    elev_rank = []
    for sat in sats:
        elev, az, dist_km = sat_altaz(sat, obs, t)
        vis = elev >= float(min_elev)
        elev_rank.append((elev, sat, az, dist_km, vis))

    elev_rank.sort(key=lambda x: x[0], reverse=True)
    elev_rank = elev_rank[: int(max_per_group.get(g, 200))]

    pts = []
    hovers = []
    for elev, sat, az, dist_km, vis in elev_rank:
        if show_only_visible and not vis:
            continue

        sat_at = sat.at(t)
        x, y, z = xyz_km_from_skyfield(sat_at)

        pts.append((x, y, z))
        hovers.append(
            f"{g} | {sat.name}"
            f"<br>ELEV {elev:.1f}°  AZ {az:.1f}°  RNG {dist_km:.0f} km"
            f"<br>VISIBLE {vis}"
        )

        if vis:
            visible_rows.append({
                "group": g,
                "name": sat.name,
                "elev": elev,
                "az": az,
                "dist_km": dist_km,
                "sat_obj": sat
            })

            # LoS
            if show_los and (not los_only_when_tracking or tracking_sat is not None):
                los_lines.append((sat.name, ground_xyz, (x, y, z), g))

    sat_points_by_group[g] = np.array(pts) if pts else np.zeros((0, 3))
    hover_by_group[g] = hovers
    top_sats_for_trails[g] = [x[1] for x in elev_rank]

# Best link
best = best_link_candidate(visible_rows)

# Trails
trails_by_group = {}
if show_trails:
    steps = max(2, int((trail_minutes * 60) // trail_step_sec) + 1)
    now_dt = t.utc_datetime()
    dt_list = [now_dt - pd.Timedelta(seconds=int((steps - 1 - i) * trail_step_sec)) for i in range(steps)]
    for g, sat_list in top_sats_for_trails.items():
        if not enabled.get(g, False):
            continue
        trails_by_group[g] = build_trails_xyz(sat_list, ts, dt_list, int(trail_top_n))

# Track cam
camera = None
title = "ORBITAL TRACKING CONSOLE — GLOBAL DISTRIBUTION"
highlight_target = None
dim_others = False

if tracking_sat is not None:
    tracking_sat_at = tracking_sat.at(t)
    tx, ty, tz = xyz_km_from_skyfield(tracking_sat_at)
    camera = compute_track_camera((tx, ty, tz), distance_factor=float(cam_distance))
    title = f"TRACK CAM — {tracking_key}"
    highlight_target = (tx, ty, tz, tracking_key)
    dim_others = bool(dim_others_on_lock)

# Next pass ground track
ground_track_xyz = None
if tracking_sat is not None:
    try:
        gt = next_pass_ground_track_xyz(ts, tracking_sat, obs, float(min_elev), lookahead_hours=int(pass_hours), samples=90)
        if gt is not None:
            ground_track_xyz = gt
    except Exception:
        ground_track_xyz = None

# Build figure
fig = make_fig(
    ground_xyz=ground_xyz,
    sat_points_by_group=sat_points_by_group,
    hover_by_group=hover_by_group,
    los_lines=los_lines if show_los else [],
    trails_by_group=trails_by_group,
    highlight_target=highlight_target,
    show_earth=show_earth,
    camera=camera,
    title=title,
    dim_others=dim_others,
    ground_track_xyz=ground_track_xyz
)

# =========================
# HUD + Tables + Pass view
# =========================
plot_ph = st.empty()
hud_ph = st.empty()
pass_ph = st.empty()
table_ph = st.empty()

with plot_ph.container():
    st.plotly_chart(fig, use_container_width=True)

# HUD lines
visible_count = len(visible_rows)
lock_line = "NO LOCK"
lock_color = HUD_RED if visible_count == 0 else HUD_GREEN

if tracking_sat is not None:
    elev_t, az_t, dist_t = sat_altaz(tracking_sat, obs, t)
    vis_t = elev_t >= float(min_elev)
    lock_line = f"LOCKED: {tracking_key} | ELEV {elev_t:.1f}° AZ {az_t:.1f}° RNG {dist_t:.0f} km | VIS {vis_t}"
    lock_color = HUD_GREEN if vis_t else HUD_RED

best_line = "BEST LINK: (none)"
best_color = HUD_RED
if best is not None:
    best_line = f"BEST LINK: {best['group']} | {best['name']} (ELEV {best['elev']:.1f}° / RNG {best['dist_km']:.0f} km)"
    best_color = HUD_GREEN

with hud_ph.container():
    st.markdown(
        f"""
        <div class="hud-card hud-mono">
          <div style="color:{lock_color}; font-size:14px;">{lock_line}</div>
          <div style="color:{best_color}; font-size:13px; margin-top:6px;">{best_line}</div>
          <div class="hud-status" style="margin-top:10px;">
            TIME(UTC) {utc_str}  |  TIME(KST) {kst_str}
            &nbsp;|&nbsp; MIN_ELEV {float(min_elev):.0f}°
            &nbsp;|&nbsp; VISIBLE {visible_count}
            &nbsp;|&nbsp; LOS {("ON" if show_los else "OFF")}
            &nbsp;|&nbsp; TRAILS {("ON" if show_trails else "OFF")}
            &nbsp;|&nbsp; LAYERS {sum(1 for k,v in enabled.items() if v)} ON
            &nbsp;|&nbsp; LIVE {("ON" if st.session_state.live else "OFF")}
          </div>
          <div class="small hud-mono" style="margin-top:8px; opacity:0.85;">
            Tip: LIVE OFF 상태에서 마우스로 회전/줌/스크롤하면 절대 튕기지 않습니다. LIVE ON은 update 간격으로 갱신됩니다.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Pass prediction
with pass_ph.container():
    st.markdown('<div class="hud-card">', unsafe_allow_html=True)
    st.markdown("### PASS PREDICTION + TIMELINE (KST)")

    if tracking_sat is None:
        st.write("TRACK TARGET을 선택하시면, 향후 패스(상승/최고점/하강)와 타임라인을 표시합니다.")
    else:
        df_pass = next_passes_table(
            ts, tracking_sat, obs,
            min_elev_deg=float(min_elev),
            hours=int(pass_hours),
            max_passes=int(pass_count)
        )
        if df_pass.empty:
            st.write(f"{pass_hours}시간 내에 MIN_ELEV={min_elev}° 기준 패스를 찾지 못했습니다.")
        else:
            st.dataframe(df_pass, use_container_width=True, hide_index=True)
            fig_tl = pass_timeline_figure(df_pass, title=f"PASS TIMELINE — {tracking_key} (KST)")
            if fig_tl is not None:
                st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Visible list
df_vis = pd.DataFrame([
    {
        "TYPE": r["group"],
        "NAME": r["name"],
        "ELEV": round(r["elev"], 1),
        "AZ": round(r["az"], 1),
        "RANGE_KM": int(r["dist_km"]),
    } for r in visible_rows
])

if not df_vis.empty:
    df_vis.sort_values("ELEV", ascending=False, inplace=True)
    df_vis = df_vis.head(60)

with table_ph.container():
    st.markdown('<div class="hud-card">', unsafe_allow_html=True)
    st.markdown("### VISIBLE TRACK LIST (TOP)")
    if df_vis.empty:
        st.write("현재 기준(최소 고도각)에서 가시 위성이 없습니다.")
    else:
        st.dataframe(df_vis, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# LIVE refresh (only if LIVE ON)
# =========================
def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            # If both fail, do nothing (prevents crash)
            pass

if st.session_state.live:
    # LIVE ON: update periodically
    time.sleep(int(update_sec))
    safe_rerun()
else:
    # LIVE OFF: no rerun (stable manual interaction)
    st.stop()
