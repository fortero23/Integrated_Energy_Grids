from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

# -----------------------------
# 0. PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

GEN_FILE = DATA_DIR / "energy-charts_Public_net_electricity_generation_in_Germany_in_2024_renewablesgene.csv"
LOAD_FILE = DATA_DIR / "energy-charts_Public_net_electricity_generation_in_Germany_in_2024_load.csv"

# -----------------------------
# 1. COMPATIBILITY
# -----------------------------
try:
    pd.options.future.infer_string = False
except Exception:
    pass

try:
    pd.options.mode.string_storage = "python"
except Exception:
    pass

# -----------------------------
# 2. HELPERS
# -----------------------------
def parse_time(series):
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except Exception:
        return pd.to_datetime(series, errors="coerce", utc=True)

def parse_number(series):
    s = series.astype(str).str.strip()
    s = s.str.replace("\u202f", "", regex=False)
    s = s.str.replace("\xa0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

# -----------------------------
# 3. READ FILES
# -----------------------------
print("Reading files...")
print("Generation file:", GEN_FILE)
print("Load file:", LOAD_FILE)

gen_raw = pd.read_csv(GEN_FILE)
load_raw = pd.read_csv(LOAD_FILE)

gen_raw.columns = [str(c).strip() for c in gen_raw.columns]
load_raw.columns = [str(c).strip() for c in load_raw.columns]

print("\nGeneration columns:")
print(gen_raw.columns.tolist())

print("\nLoad columns:")
print(load_raw.columns.tolist())

# -----------------------------
# 4. FIX COLUMN NAMES
# -----------------------------
GEN_TIME_COL = gen_raw.columns[0]
LOAD_TIME_COL = load_raw.columns[0]

# generation file expected columns
WIND_OFFSHORE_COL = "Wind offshore"
WIND_ONSHORE_COL = "Wind onshore"
SOLAR_COL = "Solar"

# load file expected column
LOAD_COL = "Load"

# -----------------------------
# 5. PARSE DATES
# -----------------------------
gen_raw[GEN_TIME_COL] = parse_time(gen_raw[GEN_TIME_COL])
load_raw[LOAD_TIME_COL] = parse_time(load_raw[LOAD_TIME_COL])

gen_raw = gen_raw.dropna(subset=[GEN_TIME_COL]).copy()
load_raw = load_raw.dropna(subset=[LOAD_TIME_COL]).copy()

gen_raw = gen_raw.set_index(GEN_TIME_COL)
load_raw = load_raw.set_index(LOAD_TIME_COL)

if getattr(gen_raw.index, "tz", None) is not None:
    gen_raw.index = gen_raw.index.tz_convert(None)

if getattr(load_raw.index, "tz", None) is not None:
    load_raw.index = load_raw.index.tz_convert(None)

# -----------------------------
# 6. CLEAN DATA
# -----------------------------
gen = pd.DataFrame(index=gen_raw.index)
gen["wind_offshore"] = parse_number(gen_raw[WIND_OFFSHORE_COL])
gen["wind_onshore"] = parse_number(gen_raw[WIND_ONSHORE_COL])
gen["solar"] = parse_number(gen_raw[SOLAR_COL])

load = pd.DataFrame(index=load_raw.index)
load["load"] = parse_number(load_raw[LOAD_COL])

df = gen.join(load, how="inner")
df = df.dropna().copy()
df = df.groupby(df.index).mean()
df = df[(df.index >= "2024-01-01") & (df.index < "2025-01-01")].copy()
df["wind"] = df["wind_onshore"] + df["wind_offshore"]
# Convert 15-minute data to hourly averages
df = df.resample("1h").mean()
print("\nFirst rows:")
print(df.head())

print("\nShape:")
print(df.shape)

# -----------------------------
# 7. CAPACITY FACTORS
# -----------------------------
WIND_CAPACITY_MW = 72700
SOLAR_CAPACITY_MW = 99300

wind_cf = (df["wind"] / WIND_CAPACITY_MW).clip(lower=0, upper=1)
solar_cf = (df["solar"] / SOLAR_CAPACITY_MW).clip(lower=0, upper=1)
demand_series = df["load"].clip(lower=0)

print("\nDemand summary:")
print(demand_series.describe())

# -----------------------------
# 8. BUILD NETWORK
# -----------------------------
n = pypsa.Network()
n.set_snapshots(df.index)

n.add("Carrier", "electricity")
n.add("Carrier", "wind")
n.add("Carrier", "solar")
n.add("Carrier", "gas")

n.add("Bus", "electricity", carrier="electricity")

n.add("Load", "demand", bus="electricity", p_set=demand_series)

n.add(
    "Generator",
    "wind",
    bus="electricity",
    carrier="wind",
    p_nom_extendable=True,
    capital_cost=130000,
    marginal_cost=0,
    p_max_pu=wind_cf
)

n.add(
    "Generator",
    "solar",
    bus="electricity",
    carrier="solar",
    p_nom_extendable=True,
    capital_cost=65000,
    marginal_cost=0,
    p_max_pu=solar_cf
)

n.add(
    "Generator",
    "gas",
    bus="electricity",
    carrier="gas",
    p_nom_extendable=True,
    capital_cost=70000,
    marginal_cost=120,
    p_max_pu=1.0
)

n.buses.index = n.buses.index.astype(object)
n.loads.index = n.loads.index.astype(object)
n.generators.index = n.generators.index.astype(object)
n.loads["bus"] = n.loads["bus"].astype(object)
n.generators["bus"] = n.generators["bus"].astype(object)

# -----------------------------
# 9. OPTIMIZE
# -----------------------------
print("\nOptimizing...")
n.optimize(solver_name="highs")

print("\nOptimal capacities (MW):")
print(n.generators.p_nom_opt)

# -----------------------------
# 10. PLOTS
# -----------------------------
summer = n.generators_t.p.loc["2024-07-01":"2024-07-07"]
summer.plot.area(figsize=(12, 5))
plt.title("Dispatch - Summer Week (Germany 2024)")
plt.ylabel("MW")
plt.tight_layout()
plt.savefig(BASE_DIR / "summer_dispatch.png", dpi=200)
plt.close()

winter = n.generators_t.p.loc["2024-01-10":"2024-01-17"]
winter.plot.area(figsize=(12, 5))
plt.title("Dispatch - Winter Week (Germany 2024)")
plt.ylabel("MW")
plt.tight_layout()
plt.savefig(BASE_DIR / "winter_dispatch.png", dpi=200)
plt.close()

annual_mix = n.generators_t.p.sum()
annual_mix.plot(kind="bar", figsize=(8, 5))
plt.title("Annual Energy Mix (Germany 2024)")
plt.ylabel("MWh")
plt.tight_layout()
plt.savefig(BASE_DIR / "annual_mix.png", dpi=200)
plt.close()

print("\nAnnual energy production (MWh):")
print(annual_mix)

capacity_factor = n.generators_t.p.sum() / (n.generators.p_nom_opt * len(n.snapshots))

print("\nCapacity factors:")
print(capacity_factor)

plt.figure(figsize=(10, 5))
for gen_name in n.generators.index:
    sorted_dispatch = n.generators_t.p[gen_name].sort_values(ascending=False).reset_index(drop=True)
    plt.plot(sorted_dispatch, label=gen_name)

plt.title("Generator Dispatch Duration Curves (Germany 2024)")
plt.xlabel("Hour rank")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "duration_curves.png", dpi=200)
plt.close()

print("\nSaved figures in Modules folder.")