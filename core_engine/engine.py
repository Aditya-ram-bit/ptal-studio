# engine.py
import os
import warnings
import zipfile

try:
    import duckdb
    import numpy as np
    import pandas as pd
    import h3
except ImportError as e:
    raise ImportError(
        f"Missing critical dependency: {e.name}. "
        "Please run 'pip install duckdb numpy pandas h3' in your terminal."
    )

import config


def _safe_cell_to_latlon(cell):
    try:
        return h3.cell_to_latlon(cell)
    except AttributeError:
        return h3.cell_to_latlng(cell)


def _read_gtfs_file(archive, filename):
    matches = [n for n in archive.namelist() if n.endswith(filename)]
    if not matches:
        return pd.DataFrame()
    try:
        return pd.read_csv(archive.open(matches[0]), dtype=str, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _validate_zip(zip_path: str) -> tuple[bool, str]:
    """
    Validates a zip is a real GTFS archive, not a downloaded HTML error page.
    Returns (is_valid, error_message).
    Handles subfolder zips (e.g. gtfs/stops.txt instead of stops.txt at root).
    """
    REQUIRED = {"stops.txt", "trips.txt", "stop_times.txt", "routes.txt"}
    try:
        if not zipfile.is_zipfile(zip_path):
            return False, "File is not a valid ZIP archive — the URL may have returned an error page."
        with zipfile.ZipFile(zip_path, "r") as arc:
            names = arc.namelist()
            # Strip subfolder prefix — find just the filenames
            basenames = {os.path.basename(n) for n in names}
            missing = REQUIRED - basenames
            if missing:
                return False, (
                    f"Missing required GTFS files: {', '.join(sorted(missing))}. "
                    f"Found: {', '.join(sorted(basenames)[:8])}{'…' if len(basenames) > 8 else ''}."
                )
        return True, ""
    except zipfile.BadZipFile:
        return False, "Corrupted or invalid ZIP file — re-download or check the URL."
    except Exception as e:
        return False, f"Could not open archive: {e}"


def _prefix_ids(df, prefix, cols):
    for col in cols:
        if col in df.columns:
            df[col] = prefix + df[col].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Agency-mode detection: returns {prefixed_agency_id -> gtfs_route_type_str}
# ---------------------------------------------------------------------------
def _detect_agency_modes(arc, prefix):
    """
    Returns dict mapping prefixed agency_id -> route_type string.
    Falls back to empty dict if agency.txt absent.
    """
    agency_df = _read_gtfs_file(arc, "agency.txt")
    if agency_df.empty or "agency_name" not in agency_df.columns:
        return {}

    result = {}
    for _, row in agency_df.iterrows():
        raw_id = str(row.get("agency_id", "")).strip()
        a_id = prefix + raw_id if raw_id else prefix
        a_name = str(row.get("agency_name", "")).lower()

        if any(kw in a_name for kw in ["metro", "namma", "subway", "mrt", "rapid transit", "underground"]):
            result[a_id] = "1"        # Metro/subway
        elif any(kw in a_name for kw in ["rail", "train", "ir ", "indian rail", "local", "commuter"]):
            result[a_id] = "2"        # Heavy rail
        elif any(kw in a_name for kw in ["ferry", "boat", "water", "cruise", "vessel"]):
            result[a_id] = "4"        # Ferry
        elif any(kw in a_name for kw in ["tram", "streetcar", "light rail", "lrt"]):
            result[a_id] = "0"        # Tram/LRT
        else:
            result[a_id] = "3"        # Default bus
    return result


def load_all_gtfs(zip_paths: list) -> dict:
    """
    Loads GTFS tables per zip, detecting modes via agency.txt when present.
    Zips without agency.txt get route_type fallback from routes.txt directly.
    Safe prefix isolation prevents ID collisions across multiple uploads.
    """
    tables = {k: [] for k in ["stops", "trips", "stop_times", "routes", "calendar", "calendar_dates", "agency", "frequencies"]}
    all_detected_modes = {}   # prefixed_agency_id -> route_type_str across all zips

    for zip_path in zip_paths:
        prefix = os.path.splitext(os.path.basename(zip_path))[0] + "_"

        with zipfile.ZipFile(zip_path, "r") as arc:
            # --- Per-zip agency detection (isolated, no cross-contamination) ---
            zip_modes = _detect_agency_modes(arc, prefix)
            all_detected_modes.update(zip_modes)

            for t_name in tables.keys():
                df = _read_gtfs_file(arc, f"{t_name}.txt")
                if not df.empty:
                    p_cols = ["stop_id", "parent_station", "trip_id", "route_id",
                              "service_id", "agency_id"]
                    tables[t_name].append(_prefix_ids(df, prefix, p_cols))

    combined_tables = {
        k: pd.concat(v, ignore_index=True) if v else pd.DataFrame()
        for k, v in tables.items()
    }
    combined_tables["_detected_modes"] = all_detected_modes
    return combined_tables


# Friendly names for error messages
_GTFS_TABLE_LABELS = {
    "stops":      "stops.txt (stop locations)",
    "trips":      "trips.txt (trip definitions)",
    "stop_times": "stop_times.txt (departure times)",
    "routes":     "routes.txt (route definitions)",
}

def compute_headways(gtfs: dict, day_start: str, day_end: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    missing_tables = [
        _GTFS_TABLE_LABELS.get(t, t)
        for t in ["stops", "trips", "stop_times", "routes"]
        if gtfs[t].empty
    ]
    if missing_tables:
        con.close()
        raise ValueError(
            "The following required GTFS files are missing or empty:\n• "
            + "\n• ".join(missing_tables)
            + "\n\nCheck that your ZIP contains a valid GTFS feed."
        )

    stops_df = gtfs["stops"].copy()
    stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
    stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")
    stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])
    stops_df["cluster_key"] = (stops_df["stop_lat"].round(4).astype(str) + "_" +
                               stops_df["stop_lon"].round(4).astype(str))

    cluster_coords = (
        stops_df.groupby("cluster_key")[["stop_lat", "stop_lon"]].mean().reset_index()
        .rename(columns={"stop_lat": "cluster_lat", "stop_lon": "cluster_lon"})
    )
    stops_df = stops_df.merge(cluster_coords, on="cluster_key", how="left")

    # Route-type resolution: agency.txt detection takes priority, then routes.txt field
    routes_df = gtfs["routes"].copy()
    detected = gtfs.get("_detected_modes", {})

    if detected and "agency_id" in routes_df.columns:
        def assign_mode(row):
            a_id = str(row.get("agency_id", "")).strip()
            if a_id in detected:
                return detected[a_id]
            rt = str(row.get("route_type", "")).strip()
            return rt if rt and rt != "nan" else "3"
        routes_df["route_type"] = routes_df.apply(assign_mode, axis=1)
    else:
        # No agency.txt in any zip — trust routes.txt route_type directly
        if "route_type" not in routes_df.columns:
            routes_df["route_type"] = "3"
        routes_df["route_type"] = routes_df["route_type"].fillna("3")

    con.register("stops_mapped", stops_df)
    con.register("trips_raw", gtfs["trips"])
    con.register("stop_times_raw", gtfs["stop_times"])
    con.register("routes_raw", routes_df)

    # ── Active services: use whatever calendar data exists ───────────────────
    # Bug fix: old code did calendar UNION calendar_dates which is correct,
    # but if calendar.txt is empty AND calendar_dates has entries, the UNION
    # of an empty string + calendar_dates query was malformed for some feeds.
    # Now: collect parts independently, join with UNION only if both exist.
    active_parts = []
    if not gtfs["calendar"].empty:
        con.register("calendar_raw", gtfs["calendar"])
        active_parts.append("SELECT DISTINCT service_id FROM calendar_raw")
    if not gtfs["calendar_dates"].empty:
        con.register("cal_dates_raw", gtfs["calendar_dates"])
        active_parts.append(
            "SELECT DISTINCT service_id FROM cal_dates_raw WHERE exception_type != '2'"
        )
    if active_parts:
        active_sql = " UNION ".join(f"({p})" for p in active_parts)
    else:
        # No calendar data at all — treat every trip as active
        active_sql = "SELECT DISTINCT service_id FROM trips_raw"

    # ── frequencies.txt: extract per-trip headway directly ───────────────────
    has_frequencies = (
        "frequencies" in gtfs
        and isinstance(gtfs["frequencies"], pd.DataFrame)
        and not gtfs["frequencies"].empty
        and "headway_secs" in gtfs["frequencies"].columns
    )
    if has_frequencies:
        con.register("frequencies_raw", gtfs["frequencies"])

    query = f"""
    WITH active_services AS ({active_sql}),
    clean_st AS (
        SELECT st.stop_id, st.trip_id,
            (CAST(SPLIT_PART(st.departure_time,':',1) AS INTEGER)*3600 +
             CAST(SPLIT_PART(st.departure_time,':',2) AS INTEGER)*60  +
             CAST(SPLIT_PART(st.departure_time,':',3) AS INTEGER)) AS dep_sec
        FROM stop_times_raw st
        WHERE st.departure_time IS NOT NULL AND LENGTH(st.departure_time) >= 7
    ),
    windowed AS (
        SELECT *, FLOOR(dep_sec/3600) AS dep_hour
        FROM clean_st
        WHERE dep_sec BETWEEN
            (CAST(SPLIT_PART('{day_start}',':',1) AS INTEGER)*3600 + CAST(SPLIT_PART('{day_start}',':',2) AS INTEGER)*60)
            AND
            (CAST(SPLIT_PART('{day_end}',':',1) AS INTEGER)*3600 + CAST(SPLIT_PART('{day_end}',':',2) AS INTEGER)*60 + 59)
    ),
    enriched AS (
        SELECT sm.cluster_key AS stop_id, w.trip_id, w.dep_sec, w.dep_hour,
               t.route_id, r.route_type,
               sm.cluster_lat AS stop_lat, sm.cluster_lon AS stop_lon
        FROM windowed w
        JOIN stops_mapped sm ON w.stop_id = sm.stop_id
        JOIN trips_raw t ON w.trip_id = t.trip_id
        JOIN routes_raw r ON t.route_id = r.route_id
        JOIN active_services a ON t.service_id = a.service_id
    ),
    gaps AS (
        SELECT stop_id, route_id, route_type, dep_hour, stop_lat, stop_lon, dep_sec,
            LAG(dep_sec) OVER (PARTITION BY stop_id, route_id, dep_hour ORDER BY dep_sec) AS prev_dep_sec
        FROM enriched
    ),
    gap_min AS (
        SELECT stop_id, route_id, route_type, dep_hour, stop_lat, stop_lon,
               (dep_sec - prev_dep_sec)/60.0 AS gap_minutes
        FROM gaps
        WHERE prev_dep_sec IS NOT NULL
          AND (dep_sec - prev_dep_sec) > 0
          AND (dep_sec - prev_dep_sec) <= 7200
    ),
    hourly AS (
        SELECT stop_id, route_id, route_type, stop_lat, stop_lon, dep_hour,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_minutes) AS hw_min,
            COUNT(*)+1 AS trip_cnt
        FROM gap_min GROUP BY stop_id, route_id, route_type, stop_lat, stop_lon, dep_hour
    ),
    best AS (
        SELECT stop_id, route_id, route_type, stop_lat, stop_lon,
               MIN(hw_min) AS headway_min, SUM(trip_cnt) AS trip_count
        FROM hourly GROUP BY stop_id, route_id, route_type, stop_lat, stop_lon
    ),
    single_trip AS (
        SELECT e.stop_id, e.route_id, e.route_type,
               AVG(CAST(e.stop_lat AS DOUBLE)) AS stop_lat,
               AVG(CAST(e.stop_lon AS DOUBLE)) AS stop_lon
        FROM enriched e
        LEFT JOIN best b ON e.stop_id = b.stop_id
        WHERE b.stop_id IS NULL
        GROUP BY e.stop_id, e.route_id, e.route_type
    )
    SELECT stop_id, route_id, CAST(route_type AS INTEGER) AS route_type,
           MIN(headway_min) AS headway_min, SUM(trip_count) AS trip_count,
           AVG(stop_lat) AS stop_lat, AVG(stop_lon) AS stop_lon
    FROM best GROUP BY stop_id, route_id, route_type
    UNION ALL
    SELECT stop_id, route_id, CAST(route_type AS INTEGER) AS route_type,
           120.0 AS headway_min, 1 AS trip_count,
           CAST(stop_lat AS DOUBLE), CAST(stop_lon AS DOUBLE)
    FROM single_trip
    """
    result = con.execute(query).df()

    # ── Merge frequencies.txt results (overrides gap estimate for same stop+route) ─
    if has_frequencies:
        freq_query = f"""
        WITH active_services AS ({active_sql}),
        freq_hw AS (
            SELECT f.trip_id,
                   AVG(CAST(f.headway_secs AS DOUBLE)) / 60.0 AS freq_hw_min
            FROM frequencies_raw f
            JOIN trips_raw t ON f.trip_id = t.trip_id
            JOIN active_services a ON t.service_id = a.service_id
            GROUP BY f.trip_id
        )
        SELECT sm.cluster_key AS stop_id, t.route_id,
               CAST(r.route_type AS INTEGER) AS route_type,
               MIN(fh.freq_hw_min) AS headway_min,
               COUNT(DISTINCT fh.trip_id) AS trip_count,
               AVG(CAST(sm.cluster_lat AS DOUBLE)) AS stop_lat,
               AVG(CAST(sm.cluster_lon AS DOUBLE)) AS stop_lon
        FROM freq_hw fh
        JOIN trips_raw t ON fh.trip_id = t.trip_id
        JOIN routes_raw r ON t.route_id = r.route_id
        JOIN stop_times_raw st ON st.trip_id = fh.trip_id
        JOIN stops_mapped sm ON st.stop_id = sm.stop_id
        GROUP BY sm.cluster_key, t.route_id, r.route_type
        """
        freq_result = con.execute(freq_query).df()
        if not freq_result.empty:
            # frequencies.txt wins over gap estimation for matching stop+route
            merge_key = ["stop_id", "route_id"]
            freq_result = freq_result[freq_result["headway_min"].notna()]
            result = result[~result.set_index(merge_key).index.isin(
                freq_result.set_index(merge_key).index
            )]
            result = pd.concat([result, freq_result], ignore_index=True)

    con.close()
    return result


def compute_ptal_matrix(hex_grid: pd.DataFrame, headways_df: pd.DataFrame, ddr: float = None) -> pd.DataFrame:
    if ddr is None:
        ddr = config.DEFAULT_DDR
    _C0 = config.PTAL_COLOR_MAP["0"] + [200]
    known = set(config.RELIABILITY_FACTORS.keys())
    hw = headways_df[headways_df["route_type"].isin(known)].copy().reset_index(drop=True)

    hex_ids = hex_grid["hex_id"].values
    poi_lats = hex_grid["poi_lat"].values.astype(float)
    poi_lons = hex_grid["poi_lon"].values.astype(float)
    n_hex = len(hex_ids)

    if hw.empty or n_hex == 0:
        return pd.DataFrame({
            "hex_id": hex_ids, "AI": np.zeros(n_hex),
            "PTAL": ["0"] * n_hex, "color_rgb": [_C0] * n_hex
        })

    s_lat = hw["stop_lat"].values.astype(float)
    s_lon = hw["stop_lon"].values.astype(float)
    s_hw  = hw["headway_min"].values.astype(float)
    s_rt  = hw["route_type"].values.astype(int)
    s_rm  = np.array([config.RELIABILITY_FACTORS.get(int(r), 0.0) for r in s_rt])
    s_wt  = np.array([config.MODE_WEIGHTS.get(int(r), 1.0) for r in s_rt])

    rail_types = {0, 1, 2, 5, 6, 7}
    bus_types  = {3, 4, 11, 12}
    is_rail = np.isin(s_rt, list(rail_types))
    is_bus  = np.isin(s_rt, list(bus_types))
    s_awt   = (s_hw / 2.0) + s_rm * s_hw

    ai_scores = np.zeros(n_hex, dtype=float)
    CHUNK = 1000

    for start in range(0, n_hex, CHUNK):
        end = min(start + CHUNK, n_hex)
        h_lat = poi_lats[start:end, None]
        h_lon = poi_lons[start:end, None]
        dlat  = np.radians(s_lat - h_lat)
        dlon  = np.radians(s_lon - h_lon)
        a     = (np.sin(dlat / 2) ** 2 +
                 np.cos(np.radians(h_lat)) * np.cos(np.radians(s_lat)) * np.sin(dlon / 2) ** 2)
        dist  = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        in_catchment = (
            (is_bus  & (dist <= config.MAX_BUS_DIST_KM)) |
            (is_rail & (dist <= config.MAX_RAIL_DIST_KM))
        )
        wt  = (dist * ddr / config.WALKING_SPEED) * 60.0
        tat = wt + s_awt
        edf = np.where(in_catchment & (tat > 0), 30.0 / tat, 0.0)

        chunk_ai = np.zeros(end - start, dtype=float)
        for mg_mask in [is_rail, is_bus]:
            if not mg_mask.any():
                continue
            mg_wt = s_wt[mg_mask][0]
            mg_sorted = np.sort(edf[:, mg_mask], axis=1)[:, ::-1]
            chunk_ai += mg_wt * (mg_sorted[:, 0] + 0.5 * mg_sorted[:, 1:].sum(axis=1))
        ai_scores[start:end] = chunk_ai

    ptal_bands = [config.ai_to_ptal(float(ai)) for ai in ai_scores]
    colors = [config.PTAL_COLOR_MAP.get(b, config.PTAL_COLOR_MAP["0"]) + [200] for b in ptal_bands]
    return pd.DataFrame({"hex_id": hex_ids, "AI": np.round(ai_scores, 3), "PTAL": ptal_bands, "color_rgb": colors})