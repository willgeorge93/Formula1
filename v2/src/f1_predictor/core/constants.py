"""F1-specific constants extracted from notebooks."""

from typing import Final

# Championship points system (current F1 rules)
POINTS_SYSTEM: Final[dict[int, int]] = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}

# Driver ID normalizations (handling special characters)
DRIVER_ID_MAPPINGS: Final[dict[str, str]] = {
    "fittipaldi": "pietro_fittipaldi",
    "räikkönen": "raikkonen",
    "pérez": "perez",
    "hülkenberg": "hulkenberg",
    "magnussen": "kevin_magnussen",
    "verstappen": "max_verstappen",
    "grosjean": "grosjean",
    "schumacher": "mick_schumacher",
}

# Race name normalizations for URL construction
RACE_NAME_MAPPINGS: Final[dict[str, str]] = {
    "mexico-city": "mexican",
    "united-states": "us",
    "usa": "us",
    "mexico": "mexican",
    "70th-anniversary": "anniversary",
}

# Constructor ID mappings for historical consistency
CONSTRUCTOR_MAPPINGS: Final[dict[str, str]] = {
    "alpine": "renault",
    "lotus_f1": "renault",
    "force_india": "racing_point",
    "aston_martin": "racing_point",
    "toro_rosso": "alphatauri",
    "marussia": "manor",
    "sauber": "alfa",
}

# Circuit ID mappings from Wikipedia names
CIRCUIT_ID_MAPPINGS: Final[dict[str, str]] = {
    "adelaide": "adelaide",
    "albert": "albert_park",
    "americas": "americas",
    "baku": "BAK",
    "bahrain": "bahrain",
    "brands hatch": "brands_hatch",
    "buddh": "buddh",
    "catalunya": "catalunya",
    "hungaroring": "hungaroring",
    "imola": "imola",
    "indianapolis": "indianapolis",
    "interlagos": "interlagos",
    "jeddah": "jeddah",
    "marina bay": "marina_bay",
    "miami": "miami",
    "monaco": "monaco",
    "monza": "monza",
    "paul ricard": "ricard",
    "red bull ring": "red_bull_ring",
    "sepang": "sepang",
    "shanghai": "shanghai",
    "silverstone": "silverstone",
    "sochi": "sochi",
    "spa": "spa",
    "suzuka": "suzuka",
    "vegas": "vegas",
    "villeneuve": "villeneuve",
    "yas marina": "yas_marina",
    "zandvoort": "zandvoort",
}

# Feature columns for the model
CATEGORICAL_FEATURES: Final[list[str]] = [
    "direction",
    "country",
    "locality",
    "type",
    "season",
    "round",
    "qual_position",
    "grid",
    "race_name",
]

NUMERICAL_FEATURES: Final[list[str]] = [
    "q_mean",
    "q_best",
    "q_worst",
    "length",
    "ageDuringRace",
]

TEXT_FEATURES: Final[list[str]] = ["weather"]

# All features for the model
ALL_FEATURES: Final[list[str]] = (
    TEXT_FEATURES + CATEGORICAL_FEATURES + NUMERICAL_FEATURES
)

# Status classifications for DNF analysis
NO_FAULT_STATUSES: Final[list[str]] = [
    "Finished",
    "+1 Lap",
    "+2 Laps",
    "+3 Laps",
    "+4 Laps",
    "+5 Laps",
    "+6 Laps",
    "+8 Laps",
]

DRIVER_FAULT_STATUSES: Final[list[str]] = [
    "Retired",
    "Withdrew",
    "Collision",
    "Accident",
    "Disqualified",
    "Damage",
    "Spun off",
    "Collision damage",
    "Puncture",
    "Rear wing",
    "Tyre",
    "Front wing",
    "Excluded",
    "Illness",
]

CAR_FAULT_STATUSES: Final[list[str]] = [
    "Suspension",
    "Wheel",
    "Vibrations",
    "Engine",
    "ERS",
    "Power loss",
    "Water leak",
    "Oil pressure",
    "Hydraulics",
    "Steering",
    "Power Unit",
    "Brakes",
    "Mechanical",
    "Turbo",
    "Battery",
    "Electrical",
    "Gearbox",
    "Wheel nut",
    "Technical",
    "Fuel system",
    "Clutch",
    "Out of fuel",
    "Driveshaft",
    "Transmission",
    "Fuel pressure",
    "Exhaust",
    "Oil leak",
    "Electronics",
    "Drivetrain",
    "Overheating",
    "Water pressure",
    "Radiator",
    "Debris",
    "Throttle",
    "Spark plugs",
    "Brake duct",
    "Seat",
]
