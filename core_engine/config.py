# config.py
WALKING_SPEED     = 4.0
DEFAULT_DDR       = 1.3
MAX_BUS_DIST_KM   = 0.500
MAX_RAIL_DIST_KM  = 2.000
DAY_START         = "05:00:00"
DAY_END           = "23:59:59"
H3_RESOLUTION     = 9
GRID_BUFFER_DIST_KM = 0.5

# Extended mode definitions to handle diverse agency.txt profiles
RELIABILITY_FACTORS = {
    0: 0.000, 1: 0.000, 2: 0.000,  # Tram, Metro, Rail
    3: 0.135, 11: 0.135,           # Bus, Trolleybus
    4: 0.200,                      # Ferry/Boats
    5: 0.100, 6: 0.100, 7: 0.100,  # Cable Car, Gondola, Funicular
    12: 0.500                      # Demand Responsive
}

MODE_WEIGHTS = {
    0: 1.0, 1: 1.0, 2: 1.0,
    3: 1.0, 11: 1.0,
    4: 1.0,                        # Ferries / Waterways
    5: 1.0, 6: 1.0, 7: 1.0,
    12: 0.25
}

PTAL_BAND_SYSTEM = "delhi"

PTAL_BANDS_DELHI = [
    (0.0,   2.0,   "1"), (2.0,   3.0,   "2"), (3.0,   5.5,   "3"),
    (5.5,   7.0,   "4"), (7.0,   8.5,   "5"), (8.5,  12.0,   "6"),
    (12.0, 20.0,   "7"), (20.0, 30.0,   "8"), (30.0, 9999.0, "9"),
]

PTAL_BANDS_TFL = [
    (0.0,   0.0,   "0"),
    (0.01,  5.0,   "1a"), (5.0,  10.0,   "1b"),
    (10.0, 15.0,   "2"),  (15.0, 20.0,   "3"),
    (20.0, 25.0,   "4"),  (25.0, 30.0,   "5"),
    (30.0, 40.0,   "6a"), (40.0, 9999.0, "6b")
]

PTAL_COLOR_MAP = {
    "0":   [18,  16,  26],    # Deep Space Black
    "1":   [180,  40,  40],   # Deep Crimson
    "1a":  [180,  40,  40],
    "1b":  [215,  48,  39],   # Strong Rust Red
    "2":   [244, 109,  67],   # Burnt Orange
    "3":   [253, 174,  97],   # Muted Amber
    "4":   [254, 224, 139],   # Pastel Yellow
    "5":   [217, 239, 139],   # Lime Tint
    "6":   [166, 217, 106],   # Light Meadow Green
    "6a":  [166, 217, 106],
    "6b":  [ 77, 172,  38],   # Deep Emerald
    "7":   [ 26, 150,  65],   # High-Velocity Green
    "8":   [ 10, 110,  45],   # Peak Core Transit Green
    "9":   [  5,  75,  30]    # Ultra Core Tier Green
}


def ai_to_ptal(ai: float) -> str:
    bands = PTAL_BANDS_DELHI if PTAL_BAND_SYSTEM == "delhi" else PTAL_BANDS_TFL
    for low, high, band in bands:
        if low <= ai < high:
            return band
    return bands[-1][2]