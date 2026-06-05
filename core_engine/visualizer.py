# visualizer.py
import os
import warnings
import json
import zipfile
import pydeck as pdk
import pandas as pd
import numpy as np
import config
import grid_gen
import engine


# ---------------------------------------------------------------------------
# Line + stop color constants
# ---------------------------------------------------------------------------
_COLOR_METRO  = [220,  30,  60]   # Red         — Metro / Subway (type 0, 1)
_COLOR_RAIL   = [140,  30, 200]   # Purple      — Heavy Rail (type 2)
_COLOR_FERRY  = [  0, 160, 220]   # Blue        — Ferry / Boat (type 4)
_COLOR_TRAM   = [  0, 190, 160]   # Teal        — Tram / LRT (type 5, 6, 7)

# Stop node fill colors per mode
_STOP_COLOR_BUS   = [  0, 100, 220, 210]   # Blue
_STOP_COLOR_METRO = [220,  30,  60, 230]   # Red
_STOP_COLOR_RAIL  = [140,  30, 200, 230]   # Purple
_STOP_COLOR_FERRY = [  0, 160, 220, 210]   # Cyan-blue
_STOP_COLOR_TRAM  = [  0, 190, 160, 210]   # Teal


def _route_type_to_line_color(route_type_str: str) -> list:
    rt = str(route_type_str).strip()
    if rt in {"0", "5", "6", "7"}:
        return _COLOR_TRAM
    if rt in {"1"}:
        return _COLOR_METRO
    if rt in {"2"}:
        return _COLOR_RAIL
    if rt in {"4"}:
        return _COLOR_FERRY
    return _COLOR_METRO   # fallback for unknown rail-ish


def _stop_color_for_type(rt: int) -> list:
    if rt in {0, 5, 6, 7}:
        return _STOP_COLOR_TRAM
    if rt == 1:
        return _STOP_COLOR_METRO
    if rt == 2:
        return _STOP_COLOR_RAIL
    if rt == 4:
        return _STOP_COLOR_FERRY
    return _STOP_COLOR_BUS


def _detect_agency_modes_vis(arc, prefix):
    """Thin copy for visualizer — avoids importing engine to prevent circular deps."""
    namelist = arc.namelist()
    agency_f = [n for n in namelist if n.endswith("agency.txt")]
    if not agency_f:
        return {}
    agency_df = pd.read_csv(arc.open(agency_f[0]), dtype=str, low_memory=False)
    if "agency_name" not in agency_df.columns:
        return {}
    result = {}
    for _, row in agency_df.iterrows():
        raw_id = str(row.get("agency_id", "")).strip()
        a_id   = prefix + raw_id if raw_id else prefix
        a_name = str(row.get("agency_name", "")).lower()
        if any(kw in a_name for kw in ["metro", "namma", "subway", "mrt", "rapid transit", "underground"]):
            result[a_id] = "1"
        elif any(kw in a_name for kw in ["rail", "train", "ir ", "indian rail", "local", "commuter"]):
            result[a_id] = "2"
        elif any(kw in a_name for kw in ["ferry", "boat", "water", "cruise", "vessel"]):
            result[a_id] = "4"
        elif any(kw in a_name for kw in ["tram", "streetcar", "light rail", "lrt"]):
            result[a_id] = "0"
        else:
            result[a_id] = "3"
    return result


def _build_paths_from_stop_sequence(arc, trips_df, valid_route_ids, stops_lookup):
    """
    Fallback when shapes.txt is absent or has no usable shape_ids.
    Reconstructs one polyline per route by picking one representative trip
    and ordering its stops by stop_sequence.
    stops_lookup: dict stop_id -> (lat, lon)
    """
    namelist = arc.namelist()
    st_f = [n for n in namelist if n.endswith("stop_times.txt")]
    if not st_f:
        return {}  # nothing we can do

    st_df = pd.read_csv(arc.open(st_f[0]), dtype=str,
                        usecols=["trip_id", "stop_id", "stop_sequence"], low_memory=False)
    st_df["stop_sequence"] = pd.to_numeric(st_df["stop_sequence"], errors="coerce")
    st_df = st_df.dropna(subset=["stop_sequence"])

    # One representative trip per route (shortest read)
    rep_trips = trips_df[trips_df["route_id"].isin(valid_route_ids)].drop_duplicates("route_id")

    route_paths = {}
    for _, t_row in rep_trips.iterrows():
        tid = t_row["trip_id"]
        rid = t_row["route_id"]
        seg = st_df[st_df["trip_id"] == tid].sort_values("stop_sequence")
        path = []
        for _, s in seg.iterrows():
            coord = stops_lookup.get(s["stop_id"])
            if coord:
                path.append([coord[1], coord[0]])  # [lon, lat] for pydeck
        if len(path) >= 2:
            route_paths[rid] = path
    return route_paths


def _load_multi_modal_lines(zip_paths: list) -> pd.DataFrame:
    """
    Returns a single DataFrame of all non-bus line geometries with mode-specific colors.
    Primary: shapes.txt  →  Fallback: stop_sequence reconstruction
    Agency.txt overrides route_type per zip if present.
    """
    non_bus_types = {"0", "1", "2", "4", "5", "6", "7"}
    all_rows = []

    for zip_path in zip_paths:
        prefix = os.path.splitext(os.path.basename(zip_path))[0] + "_"

        with zipfile.ZipFile(zip_path, "r") as arc:
            namelist = arc.namelist()
            route_f  = [n for n in namelist if n.endswith("routes.txt")]
            shape_f  = [n for n in namelist if n.endswith("shapes.txt")]
            trips_f  = [n for n in namelist if n.endswith("trips.txt")]
            stops_f  = [n for n in namelist if n.endswith("stops.txt")]

            if not route_f or not trips_f:
                continue

            routes_df = pd.read_csv(arc.open(route_f[0]), dtype=str, low_memory=False)
            routes_df["route_id"] = prefix + routes_df["route_id"].astype(str)

            # Agency override per zip
            zip_modes = _detect_agency_modes_vis(arc, prefix)
            if zip_modes and "agency_id" in routes_df.columns:
                routes_df["agency_id_p"] = prefix + routes_df["agency_id"].astype(str)
                routes_df["route_type"] = routes_df.apply(
                    lambda r: zip_modes.get(r["agency_id_p"], r.get("route_type", "3")), axis=1
                )

            valid_routes = routes_df[routes_df["route_type"].isin(non_bus_types)].copy()
            if valid_routes.empty:
                continue

            trips_df  = pd.read_csv(arc.open(trips_f[0]), dtype=str, low_memory=False)
            trips_df["route_id"] = prefix + trips_df["route_id"].astype(str)

            # ── Try shapes.txt first ──────────────────────────────────────────
            use_shapes = False
            shape_paths = {}

            if shape_f:
                shapes_df = pd.read_csv(arc.open(shape_f[0]), dtype=str, low_memory=False)
                shapes_df["shape_pt_lat"]      = pd.to_numeric(shapes_df["shape_pt_lat"],      errors="coerce")
                shapes_df["shape_pt_lon"]      = pd.to_numeric(shapes_df["shape_pt_lon"],      errors="coerce")
                shapes_df["shape_pt_sequence"] = pd.to_numeric(shapes_df["shape_pt_sequence"], errors="coerce")
                shapes_df = shapes_df.dropna().sort_values(["shape_id", "shape_pt_sequence"])

                if not shapes_df.empty and "shape_id" in trips_df.columns:
                    trips_df["shape_id"] = trips_df["shape_id"].fillna("")
                    has_shape_ids = trips_df["shape_id"].str.strip().ne("").any()

                    if has_shape_ids:
                        shape_paths = {
                            sid: [[float(r.shape_pt_lon), float(r.shape_pt_lat)]
                                  for r in grp.itertuples()]
                            for sid, grp in shapes_df.groupby("shape_id")
                        }
                        use_shapes = True

            # ── Fallback: stop_sequence reconstruction ────────────────────────
            if not use_shapes:
                stops_lookup = {}
                if stops_f:
                    s_df = pd.read_csv(arc.open(stops_f[0]), dtype=str, low_memory=False)
                    s_df["stop_lat"] = pd.to_numeric(s_df["stop_lat"], errors="coerce")
                    s_df["stop_lon"] = pd.to_numeric(s_df["stop_lon"], errors="coerce")
                    s_df = s_df.dropna(subset=["stop_lat", "stop_lon"])
                    stops_lookup = {
                        prefix + str(r.stop_id): (float(r.stop_lat), float(r.stop_lon))
                        for r in s_df.itertuples()
                    }
                # trips_df already has prefixed route_id; stop_times stop_id needs prefix too
                trips_for_fallback = trips_df.copy()
                trips_for_fallback["trip_id_orig"] = trips_for_fallback["trip_id"]
                valid_route_ids = set(valid_routes["route_id"])
                route_paths_fallback = _build_paths_from_stop_sequence(
                    arc, trips_for_fallback, valid_route_ids, stops_lookup
                )

            # ── Emit rows ─────────────────────────────────────────────────────
            for _, r_row in valid_routes.iterrows():
                r_id   = r_row["route_id"]
                r_type = str(r_row["route_type"])
                color  = _route_type_to_line_color(r_type)

                raw_hex = str(r_row.get("route_color", "")).strip().lstrip("#")
                if len(raw_hex) == 6:
                    try:
                        color = [int(raw_hex[i:i+2], 16) for i in (0, 2, 4)]
                    except Exception:
                        pass

                if use_shapes:
                    matched = (
                        trips_df[trips_df["route_id"] == r_id]
                        .drop_duplicates("shape_id")[["shape_id"]]
                    )
                    for _, t_row in matched.iterrows():
                        path = shape_paths.get(t_row["shape_id"])
                        if path:
                            all_rows.append({"path": path, "color": color,
                                             "name": r_row.get("route_long_name", r_id), "r_type": r_type})
                else:
                    path = route_paths_fallback.get(r_id)
                    if path:
                        all_rows.append({"path": path, "color": color,
                                         "name": r_row.get("route_long_name", r_id), "r_type": r_type})

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=["path", "color", "name", "r_type"])


_OVERLAY_HTML = """
<style>
#ptal-ctrl { position:absolute; top:16px; right:16px; background:rgba(30,41,59,0.95); border:1px solid #475569; border-radius:10px; padding:14px 18px; font-family:-apple-system,sans-serif; font-size:13px; color:#F8F9FA; z-index:99999; min-width:230px; box-shadow:0 4px 12px rgba(0,0,0,0.3); }
#ptal-ctrl .ctrl-title { font-size:11px; font-weight:700; text-transform:uppercase; color:#94A3B8; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #334155; }
#ptal-ctrl label { display:flex; align-items:center; gap:10px; cursor:pointer; margin-bottom:7px; }
#ptal-ctrl input[type=checkbox] { width:15px; height:15px; accent-color:#38BDF8; cursor:pointer; }
.ctrl-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
#ptal-legend { position:absolute; bottom:30px; left:16px; background:rgba(30,41,59,0.95); border:1px solid #475569; border-radius:10px; padding:12px 16px; font-family:-apple-system,sans-serif; font-size:11px; color:#F8F9FA; z-index:99999; box-shadow:0 4px 12px rgba(0,0,0,0.3); }
.leg-row { display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.leg-sw { width:13px; height:13px; border-radius:2px; flex-shrink:0; }
</style>
<div id="ptal-ctrl">
  <div class="ctrl-title">Layer Controls</div>
  <label><input type="checkbox" id="chk-hex" checked> PTAL Hex Grid</label>
  <label><input type="checkbox" id="chk-bus" checked><span class="ctrl-dot" style="background:#0064DC"></span> Bus Stop Nodes</label>
  <label><input type="checkbox" id="chk-metro-stops" checked><span class="ctrl-dot" style="background:#DC1E3C"></span> Metro Stop Nodes</label>
  <label><input type="checkbox" id="chk-rail-stops" checked><span class="ctrl-dot" style="background:#8C1EC8"></span> Rail Stop Nodes</label>
  <label><input type="checkbox" id="chk-metro" checked><span style="display:inline-block;width:20px;height:3px;background:#DC1E3C;border-radius:2px"></span> Metro Lines</label>
  <label><input type="checkbox" id="chk-rail" checked><span style="display:inline-block;width:20px;height:3px;background:#8C1EC8;border-radius:2px"></span> Rail Lines</label>
  <label><input type="checkbox" id="chk-ferry" checked><span style="display:inline-block;width:20px;height:3px;background:#00A0DC;border-radius:2px"></span> Ferry/Waterways</label>
  <label><input type="checkbox" id="chk-tram" checked><span style="display:inline-block;width:20px;height:3px;background:#00BEA0;border-radius:2px"></span> Tram / LRT</label>
</div>
<div id="ptal-legend">
  <div class="leg-row"><div class="leg-sw" style="background:#780404"></div> 0 — No access</div>
  <div class="leg-row"><div class="leg-sw" style="background:#E60000"></div> 1 — Very poor</div>
  <div class="leg-row"><div class="leg-sw" style="background:#FF4500"></div> 2</div>
  <div class="leg-row"><div class="leg-sw" style="background:#FF8C00"></div> 3</div>
  <div class="leg-row"><div class="leg-sw" style="background:#FFD700"></div> 4</div>
  <div class="leg-row"><div class="leg-sw" style="background:#EAEA00"></div> 5</div>
  <div class="leg-row"><div class="leg-sw" style="background:#9BE100"></div> 6</div>
  <div class="leg-row"><div class="leg-sw" style="background:#32CD32"></div> 7</div>
  <div class="leg-row"><div class="leg-sw" style="background:#00BD00"></div> 8</div>
  <div class="leg-row"><div class="leg-sw" style="background:#005A00"></div> 9 — Best</div>
</div>
<script>
(function() {
  function waitForDeck() {
    var inst = window.__ptal_deck;
    if (!inst || !inst.props || !inst.props.layers || inst.props.layers.length === 0) {
      setTimeout(waitForDeck, 150); return;
    }
    var baseLayers = inst.props.layers.slice();
    function applyVis() {
      var v = {
        hex:         document.getElementById('chk-hex').checked,
        bus:         document.getElementById('chk-bus').checked,
        metroStops:  document.getElementById('chk-metro-stops').checked,
        railStops:   document.getElementById('chk-rail-stops').checked,
        metro:       document.getElementById('chk-metro').checked,
        rail:        document.getElementById('chk-rail').checked,
        ferry:       document.getElementById('chk-ferry').checked,
        tram:        document.getElementById('chk-tram').checked
      };
      inst.setProps({ layers: baseLayers.map(function(l) {
        if (l.id === 'ptal_hex')      return l.clone({ visible: v.hex });
        if (l.id === 'bus_stops')     return l.clone({ visible: v.bus });
        if (l.id === 'metro_stops')   return l.clone({ visible: v.metroStops });
        if (l.id === 'rail_stops')    return l.clone({ visible: v.railStops });
        if (l.id === 'metro_lines')   return l.clone({ visible: v.metro });
        if (l.id === 'rail_lines')    return l.clone({ visible: v.rail });
        if (l.id === 'ferry_lines')   return l.clone({ visible: v.ferry });
        if (l.id === 'tram_lines')    return l.clone({ visible: v.tram });
        return l;
      })});
    }
    ['chk-hex','chk-bus','chk-metro-stops','chk-rail-stops',
     'chk-metro','chk-rail','chk-ferry','chk-tram'].forEach(function(id) {
      document.getElementById(id).addEventListener('change', applyVis);
    });
  }
  setTimeout(waitForDeck, 300);
})();
</script>
"""


def _inject_overlay(html_path: str):
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        "const deckInstance = createDeck({",
        "window.__ptal_deck = null;\n    const deckInstance = createDeck({"
    )
    content = content.replace(
        "          });\n\n  </script>",
        "          });\n    window.__ptal_deck = deckInstance;\n\n  </script>"
    )
    content = content.replace("</body>", _OVERLAY_HTML + "\n</body>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)



def load_geojson_layers(geojson_paths: list) -> list:
    """
    Converts GeoJSON files into PyDeck layers.
    Supports Point, LineString, Polygon, and MultiPolygon geometries.
    Each file gets its own layer with a distinct pastel color.
    """
    _GJ_COLORS = [
        [255, 180,  80, 180],  # amber
        [ 80, 200, 160, 180],  # teal
        [200,  80, 200, 180],  # magenta
        [ 80, 140, 255, 180],  # blue
        [255, 100, 100, 180],  # coral
    ]
    layers = []

    for i, path in enumerate(geojson_paths):
        color = _GJ_COLORS[i % len(_GJ_COLORS)]
        fname = os.path.basename(path)
        layer_id = f"geojson_{i}"

        try:
            with open(path, "r", encoding="utf-8") as f:
                gj = json.load(f)
        except Exception as e:
            warnings.warn(f"Could not load GeoJSON {fname}: {e}")
            continue

        features = gj.get("features", []) if gj.get("type") == "FeatureCollection" else [gj]
        if not features:
            continue

        # Separate by geometry type
        points, lines, polys = [], [], []

        for feat in features:
            geom = feat.get("geometry") or feat
            props = feat.get("properties", {}) or {}
            if not geom:
                continue
            gt = geom.get("type", "")
            coords = geom.get("coordinates", [])

            if gt == "Point":
                points.append({
                    "position": [coords[0], coords[1]],
                    "name": props.get("name", props.get("NAME", fname))
                })
            elif gt == "MultiPoint":
                for c in coords:
                    points.append({"position": [c[0], c[1]], "name": props.get("name", fname)})
            elif gt == "LineString":
                lines.append({"path": [[c[0], c[1]] for c in coords], "name": props.get("name", fname)})
            elif gt == "MultiLineString":
                for line in coords:
                    lines.append({"path": [[c[0], c[1]] for c in line], "name": props.get("name", fname)})
            elif gt == "Polygon":
                polys.append({"polygon": [[c[0], c[1]] for c in coords[0]], "name": props.get("name", fname)})
            elif gt == "MultiPolygon":
                for poly in coords:
                    polys.append({"polygon": [[c[0], c[1]] for c in poly[0]], "name": props.get("name", fname)})

        if points:
            layers.append(pdk.Layer(
                "ScatterplotLayer", pd.DataFrame(points),
                id=f"{layer_id}_pts", pickable=True,
                radius_min_pixels=4, radius_max_pixels=10,
                get_position="position",
                get_fill_color=color,
                get_line_color=[255, 255, 255, 150],
                line_width_min_pixels=1
            ))
        if lines:
            layers.append(pdk.Layer(
                "PathLayer", pd.DataFrame(lines),
                id=f"{layer_id}_lines", pickable=True,
                width_min_pixels=2, get_path="path",
                get_color=color[:3], get_width=3
            ))
        if polys:
            layers.append(pdk.Layer(
                "PolygonLayer", pd.DataFrame(polys),
                id=f"{layer_id}_polys", pickable=True,
                stroked=True, filled=True, extruded=False,
                get_polygon="polygon",
                get_fill_color=color,
                get_line_color=[255, 255, 255, 180],
                line_width_min_pixels=1
            ))

    return layers

def render_ptal_map(zip_paths: list, output_path: str = "ptal_map.html",
                    place_name: str = None, ddr: float = None,
                    day_start: str = None, day_end: str = None,
                    geojson_paths: list = None):
    if ddr is None:       ddr       = config.DEFAULT_DDR
    if day_start is None: day_start = config.DAY_START
    if day_end is None:   day_end   = config.DAY_END

    geojson_only = not zip_paths  # GeoJSON-only mode — skip GTFS engine entirely

    if not geojson_only:
        gtfs        = engine.load_all_gtfs(zip_paths)
        headways_df = engine.compute_headways(gtfs, day_start, day_end)
        stops_gdf, hex_grid = grid_gen.build_grid(zip_paths, place_name=place_name)
        ptal_df     = engine.compute_ptal_matrix(hex_grid, headways_df, ddr=ddr)
    else:
        stops_gdf = None
        ptal_df   = None
        hex_grid  = None

    if not geojson_only:
        # Fill any missing hexes with zero
        scored_ids = set(ptal_df["hex_id"]) if not ptal_df.empty else set()
        missing    = [h for h in hex_grid["hex_id"].unique() if h not in scored_ids]
        if missing:
            ptal_df = pd.concat([ptal_df, pd.DataFrame({
                "hex_id":    missing,
                "AI":        [0.0] * len(missing),
                "PTAL":      ["0"] * len(missing),
                "color_rgb": [config.PTAL_COLOR_MAP["0"] + [200]] * len(missing)
            })], ignore_index=True)

        lines_df = _load_multi_modal_lines(zip_paths)

        stops_gdf["stop_lat"] = stops_gdf["stop_lat"].astype(float)
        stops_gdf["stop_lon"] = stops_gdf["stop_lon"].astype(float)

        rt_numeric = (
            pd.to_numeric(stops_gdf.get("route_type", pd.Series(["3"] * len(stops_gdf))),
                          errors="coerce").fillna(3).astype(int)
        )
        bus_mask   = rt_numeric.isin({3, 11, 12})
        metro_mask = rt_numeric.isin({1})
        rail_mask  = rt_numeric.isin({2})

        def _stop_plot(mask, color):
            df = stops_gdf[mask][["stop_lat", "stop_lon"]].copy()
            df["color"] = [color] * len(df)
            return df

        bus_plot   = _stop_plot(bus_mask,   _STOP_COLOR_BUS)
        metro_plot = _stop_plot(metro_mask, _STOP_COLOR_METRO)
        rail_plot  = _stop_plot(rail_mask,  _STOP_COLOR_RAIL)

        view = pdk.ViewState(
            latitude=float(stops_gdf["stop_lat"].mean()),
            longitude=float(stops_gdf["stop_lon"].mean()),
            zoom=11
        )
    else:
        lines_df = pd.DataFrame()
        bus_plot = metro_plot = rail_plot = pd.DataFrame()
        # Center view on GeoJSON data
        view = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=5)

    layers = []

    if not geojson_only:
        layers += [
            pdk.Layer(
                "H3HexagonLayer", ptal_df, id="ptal_hex",
                pickable=True, stroked=True, filled=True, extruded=False,
                get_hexagon="hex_id", get_fill_color="color_rgb",
                get_line_color=[200, 200, 200, 15], line_width_min_pixels=0.4, opacity=0.80
            ),
            pdk.Layer(
                "ScatterplotLayer", bus_plot, id="bus_stops",
                pickable=True, radius_min_pixels=2, radius_max_pixels=5,
                get_position="[stop_lon, stop_lat]",
                get_fill_color=_STOP_COLOR_BUS,
                get_line_color=[255, 255, 255, 120], line_width_min_pixels=0.5
            ),
        ]

    if not metro_plot.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer", metro_plot, id="metro_stops",
            pickable=True, radius_min_pixels=3, radius_max_pixels=7,
            get_position="[stop_lon, stop_lat]",
            get_fill_color=_STOP_COLOR_METRO,
            get_line_color=[255, 255, 255, 150], line_width_min_pixels=0.8
        ))

    if not rail_plot.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer", rail_plot, id="rail_stops",
            pickable=True, radius_min_pixels=3, radius_max_pixels=7,
            get_position="[stop_lon, stop_lat]",
            get_fill_color=_STOP_COLOR_RAIL,
            get_line_color=[255, 255, 255, 150], line_width_min_pixels=0.8
        ))

    # Line layers split by type
    if not lines_df.empty:
        metro_lines = lines_df[lines_df["r_type"].isin(["0", "1"])]
        rail_lines  = lines_df[lines_df["r_type"] == "2"]
        ferry_lines = lines_df[lines_df["r_type"] == "4"]
        tram_lines  = lines_df[lines_df["r_type"].isin(["5", "6", "7"])]

        if not metro_lines.empty:
            layers.append(pdk.Layer(
                "PathLayer", metro_lines, id="metro_lines",
                pickable=True, width_min_pixels=2, get_path="path",
                get_color="color", get_width=4
            ))
        if not rail_lines.empty:
            layers.append(pdk.Layer(
                "PathLayer", rail_lines, id="rail_lines",
                pickable=True, width_min_pixels=2, get_path="path",
                get_color="color", get_width=4
            ))
        if not ferry_lines.empty:
            layers.append(pdk.Layer(
                "PathLayer", ferry_lines, id="ferry_lines",
                pickable=True, width_min_pixels=3, get_path="path",
                get_color="color", get_width=5
            ))
        if not tram_lines.empty:
            layers.append(pdk.Layer(
                "PathLayer", tram_lines, id="tram_lines",
                pickable=True, width_min_pixels=2, get_path="path",
                get_color="color", get_width=3
            ))

    # GeoJSON overlay layers
    if geojson_paths:
        layers.extend(load_geojson_layers(geojson_paths))

    deck = pdk.Deck(
        layers=layers, initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": "<div style='font-family:sans-serif;font-size:12px;background:#fff;color:#222;"
                    "padding:8px;border-radius:6px;'><b>AI Score:</b> {AI}<br/><b>PTAL:</b> {PTAL}</div>"
        }
    )
    deck.to_html(output_path, open_browser=False)
    _inject_overlay(output_path)
    return os.path.abspath(output_path)