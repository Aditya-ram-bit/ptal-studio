# grid_gen.py
import warnings
import zipfile
import os
import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, MultiPoint
import config

MAX_HEX_COUNT = 150_000
_RES_FALLBACK = [9, 8, 7]


def _safe_cell_to_latlon(cell):
    try:
        return h3.cell_to_latlon(cell)
    except AttributeError:
        return h3.cell_to_latlng(cell)


def _detect_agency_modes_local(arc, prefix):
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


def load_combined_stops_with_modes(zip_paths: list) -> gpd.GeoDataFrame:
    frames = []
    for zip_path in zip_paths:
        prefix = os.path.splitext(os.path.basename(zip_path))[0] + "_"
        with zipfile.ZipFile(zip_path, "r") as arc:
            namelist = arc.namelist()
            stops_f  = [n for n in namelist if n.endswith("stops.txt")]
            routes_f = [n for n in namelist if n.endswith("routes.txt")]
            trips_f  = [n for n in namelist if n.endswith("trips.txt")]
            st_f     = [n for n in namelist if n.endswith("stop_times.txt")]
            if not stops_f:
                continue
            stops_df = pd.read_csv(arc.open(stops_f[0]), dtype=str, low_memory=False)
            stops_df["stop_id"] = prefix + stops_df["stop_id"].astype(str)

            zip_agency_modes = _detect_agency_modes_local(arc, prefix)
            has_agency_info  = bool(zip_agency_modes)

            route_mode_map = {}
            if routes_f:
                r_df = pd.read_csv(arc.open(routes_f[0]), dtype=str, low_memory=False)
                if "route_type" in r_df.columns:
                    route_mode_map = dict(zip(prefix + r_df["route_id"], r_df["route_type"]))

            if has_agency_info and routes_f:
                r_df = pd.read_csv(arc.open(routes_f[0]), dtype=str, low_memory=False)
                if "agency_id" in r_df.columns and "route_id" in r_df.columns:
                    r_df["route_id_p"]    = prefix + r_df["route_id"].astype(str)
                    r_df["agency_id_p"]   = prefix + r_df["agency_id"].astype(str)
                    r_df["resolved_type"] = r_df["agency_id_p"].map(zip_agency_modes).fillna(
                        r_df.get("route_type", "3"))
                    route_mode_map.update(dict(zip(r_df["route_id_p"], r_df["resolved_type"])))

            if trips_f and st_f and route_mode_map:
                try:
                    t_df  = pd.read_csv(arc.open(trips_f[0]),  dtype=str, usecols=["trip_id", "route_id"])
                    st_df = pd.read_csv(arc.open(st_f[0]),     dtype=str, usecols=["trip_id", "stop_id"])
                    t_df["route_id"]  = prefix + t_df["route_id"]
                    st_df["stop_id"]  = prefix + st_df["stop_id"]
                    merged_st = st_df.merge(t_df, on="trip_id")
                    merged_st["route_type"] = merged_st["route_id"].map(route_mode_map)
                    stop_mode = merged_st.groupby("stop_id")["route_type"].first().to_dict()
                    stops_df["route_type"] = stops_df["stop_id"].map(stop_mode).fillna("3")
                except Exception:
                    stops_df["route_type"] = "3"
            else:
                stops_df["route_type"] = "3"

            frames.append(stops_df)

    if not frames:
        raise FileNotFoundError("No valid GTFS stop assets found in uploaded archives.")

    merged = pd.concat(frames, ignore_index=True)
    merged["stop_lat"] = pd.to_numeric(merged["stop_lat"], errors="coerce")
    merged["stop_lon"] = pd.to_numeric(merged["stop_lon"], errors="coerce")
    merged = merged.dropna(subset=["stop_lat", "stop_lon"])
    merged["route_type"] = merged["route_type"].fillna("3")
    geom = [Point(r.stop_lon, r.stop_lat) for r in merged.itertuples()]
    return gpd.GeoDataFrame(merged, geometry=geom, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Convex hull footprint per zip
# ---------------------------------------------------------------------------

def _hexes_for_stops(lats, lons, pad_m, scale, res) -> set:
    """Tight convex hull around one zip's stops, buffered, then H3 filled."""
    if len(lats) < 3:
        hexes = set()
        for lat, lon in zip(lats, lons):
            try:
                hexes.add(h3.latlng_to_cell(float(lat), float(lon), res))
            except Exception:
                pass
        return hexes

    hull       = MultiPoint(list(zip(lons, lats))).convex_hull
    hull_proj  = gpd.GeoSeries([hull], crs="EPSG:4326").to_crs(epsg=3395).iloc[0]
    hull_proj  = hull_proj.buffer(pad_m / scale)
    hull_wgs84 = gpd.GeoSeries([hull_proj], crs="EPSG:3395").to_crs(epsg=4326).iloc[0]

    polygons = list(hull_wgs84.geoms) if isinstance(hull_wgs84, MultiPolygon) else [hull_wgs84]
    hexes = set()
    for poly in polygons:
        try:
            hexes.update(h3.geo_to_cells(poly.__geo_interface__, res))
        except Exception:
            pass
    return hexes


# ---------------------------------------------------------------------------
# Main grid builder
# ---------------------------------------------------------------------------

def build_grid(zip_paths: list, place_name: str = None) -> tuple:
    stops_gdf = load_combined_stops_with_modes(zip_paths)
    pad_m     = getattr(config, "GRID_BUFFER_DIST_KM", 0.5) * 1000.0
    base_res  = config.H3_RESOLUTION
    all_hexes = set()

    # Per-zip independent footprint — no cross-file bridging
    for zip_path in zip_paths:
        prefix    = os.path.splitext(os.path.basename(zip_path))[0] + "_"
        zip_stops = stops_gdf[stops_gdf["stop_id"].str.startswith(prefix)]
        if zip_stops.empty:
            continue

        lats  = zip_stops["stop_lat"].astype(float).values
        lons  = zip_stops["stop_lon"].astype(float).values
        scale = 1.0 / np.cos(np.radians(float(lats.mean())))

        for res in _RES_FALLBACK:
            if res > base_res:
                continue
            zip_hexes = _hexes_for_stops(lats, lons, pad_m, scale, res)

            if len(zip_hexes) <= MAX_HEX_COUNT:
                all_hexes.update(zip_hexes)
                if res < base_res:
                    warnings.warn(
                        f"{prefix}: H3 res auto-downgraded {base_res}→{res} "
                        f"({len(zip_hexes):,} hexes).", stacklevel=2
                    )
                break
            if res == _RES_FALLBACK[-1]:
                all_hexes.update(zip_hexes)

    rows = []
    for h_id in all_hexes:
        lat, lon = _safe_cell_to_latlon(h_id)
        rows.append({"hex_id": h_id, "poi_lat": lat, "poi_lon": lon})

    return stops_gdf, pd.DataFrame(rows)

