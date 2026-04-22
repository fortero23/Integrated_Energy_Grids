#!/usr/bin/env python
# coding: utf-8

import math
import re
import pandas as pd
import requests

API_BASE = "https://api.energy-charts.info"
START_2024 = "2024-01-01"
END_2024 = "2024-12-31"

COUNTRY_CODES = {
    "Germany": "de",
    "Belgium": "be",
    "France": "fr",
    "Netherlands": "nl",
}

NETWORK_EDGES = [
    ("Germany", "Belgium"),
    ("Germany", "France"),
    ("Germany", "Netherlands"),
    ("Belgium", "France"),
    ("Belgium", "Netherlands"),
]


def slugify(text):
    text = text.lower().strip()
    text = text.replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def ec_get(endpoint, **params):
    url = f"{API_BASE}/{endpoint}"
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def get_public_power_df(country_code, start=START_2024, end=END_2024):
    data_json = ec_get("public_power", country=country_code, start=start, end=end)

    timestamps = pd.to_datetime(data_json["unix_seconds"], unit="s", utc=True)
    data = {}

    for item in data_json["production_types"]:
        data[slugify(item["name"])] = item["data"]

    df = pd.DataFrame(data, index=timestamps).sort_index()
    df = df.resample("1h").mean()
    df.index = df.index.tz_convert(None)
    return df


def get_installed_power_2024(country_code):
    data_json = ec_get("installed_power", country=country_code, time_step="yearly")

    years = [str(t) for t in data_json["time"]]
    year_position = years.index("2024")

    values = {}
    for item in data_json["production_types"]:
        values[slugify(item["name"])] = item["data"][year_position]

    series = pd.Series(values, dtype=float)
    return series * 1000.0


def get_cbpf_df(country_code, start=START_2024, end=END_2024):
    data_json = ec_get("cbpf", country=country_code, start=start, end=end)

    timestamps = pd.to_datetime(data_json["unix_seconds"], unit="s", utc=True)
    data = {}

    for item in data_json["countries"]:
        data[slugify(item["name"])] = item["data"]

    df = pd.DataFrame(data, index=timestamps).sort_index()
    df = df.resample("1h").mean()
    df.index = df.index.tz_convert(None)
    return df


def find_best_matching_key(keys, patterns):
    slugged_keys = {slugify(k): k for k in keys}

    for pattern in patterns:
        pattern_slug = slugify(pattern)
        if pattern_slug in slugged_keys:
            return slugged_keys[pattern_slug]

    for pattern in patterns:
        pattern_slug = slugify(pattern)
        for key_slug, original_key in slugged_keys.items():
            if pattern_slug in key_slug:
                return original_key

    return None


def extract_installed_capacities(installed_power_series):
    keys = list(installed_power_series.index)

    mapping = {
        "wind_onshore": ["wind_onshore", "onshore"],
        "wind_offshore": ["wind_offshore", "offshore"],
        "solar": ["solar", "pv", "photovoltaic", "photovoltaics"],
        "gas": ["fossil_gas", "gas"],
    }

    capacities = {}
    for tech, patterns in mapping.items():
        key = find_best_matching_key(keys, patterns)
        capacities[tech] = float(installed_power_series[key]) if key is not None else 0.0

    return capacities


def extract_generation_bundle(public_power_df):
    columns = list(public_power_df.columns)

    load_key = find_best_matching_key(columns, ["load"])
    if load_key is None:
        raise ValueError(f"Load column not found. Available columns: {columns}")

    solar_key = find_best_matching_key(columns, ["solar", "pv", "photovoltaic", "photovoltaics"])
    wind_onshore_key = find_best_matching_key(columns, ["wind_onshore", "onshore"])
    wind_offshore_key = find_best_matching_key(columns, ["wind_offshore", "offshore"])
    gas_key = find_best_matching_key(columns, ["fossil_gas", "gas"])

    load = pd.to_numeric(public_power_df[load_key], errors="coerce").fillna(0.0)
    solar = pd.to_numeric(public_power_df[solar_key], errors="coerce").fillna(0.0) if solar_key else pd.Series(0.0, index=public_power_df.index)
    wind_onshore = pd.to_numeric(public_power_df[wind_onshore_key], errors="coerce").fillna(0.0) if wind_onshore_key else pd.Series(0.0, index=public_power_df.index)
    wind_offshore = pd.to_numeric(public_power_df[wind_offshore_key], errors="coerce").fillna(0.0) if wind_offshore_key else pd.Series(0.0, index=public_power_df.index)
    gas = pd.to_numeric(public_power_df[gas_key], errors="coerce").fillna(0.0) if gas_key else pd.Series(0.0, index=public_power_df.index)

    generation_columns = [col for col in public_power_df.columns if col != load_key]
    total_generation = public_power_df[generation_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)

    residual_generation = (total_generation - solar - wind_onshore - wind_offshore - gas).clip(lower=0.0)

    return {
        "load": load.clip(lower=0.0),
        "solar": solar.clip(lower=0.0),
        "wind_onshore": wind_onshore.clip(lower=0.0),
        "wind_offshore": wind_offshore.clip(lower=0.0),
        "gas": gas.clip(lower=0.0),
        "residual_generation": residual_generation,
    }


def get_observed_line_capacity_mw(country_a, country_b):
    code_a = COUNTRY_CODES[country_a]
    code_b = COUNTRY_CODES[country_b]

    df_a = get_cbpf_df(code_a)
    col_a = find_best_matching_key(df_a.columns, [country_b])

    if col_a is not None:
        return math.ceil(float(df_a[col_a].abs().max()) * 1000.0)

    df_b = get_cbpf_df(code_b)
    col_b = find_best_matching_key(df_b.columns, [country_a])

    if col_b is not None:
        return math.ceil(float(df_b[col_b].abs().max()) * 1000.0)

    raise ValueError(f"No cross-border flow series found for {country_a} and {country_b}")


def build_country_data_and_lines():
    country_data = {}

    for country_name, country_code in COUNTRY_CODES.items():
        public_power = get_public_power_df(country_code)
        installed_power = get_installed_power_2024(country_code)

        observed = extract_generation_bundle(public_power)
        capacities_mw = extract_installed_capacities(installed_power)

        cf_onshore = (
            observed["wind_onshore"] / capacities_mw["wind_onshore"]
            if capacities_mw["wind_onshore"] > 0
            else pd.Series(0.0, index=public_power.index)
        )

        cf_offshore = (
            observed["wind_offshore"] / capacities_mw["wind_offshore"]
            if capacities_mw["wind_offshore"] > 0
            else pd.Series(0.0, index=public_power.index)
        )

        cf_solar = (
            observed["solar"] / capacities_mw["solar"]
            if capacities_mw["solar"] > 0
            else pd.Series(0.0, index=public_power.index)
        )

        country_data[country_name] = {
            "load": observed["load"],
            "gas_dispatch": observed["gas"],
            "residual_generation": observed["residual_generation"],
            "cf_onshore": cf_onshore.clip(lower=0.0, upper=1.0),
            "cf_offshore": cf_offshore.clip(lower=0.0, upper=1.0),
            "cf_solar": cf_solar.clip(lower=0.0, upper=1.0),
            "capacities_mw": capacities_mw,
        }

    snapshots = country_data["Germany"]["load"].index

    for country_name in country_data:
        for variable_name in [
            "load",
            "gas_dispatch",
            "residual_generation",
            "cf_onshore",
            "cf_offshore",
            "cf_solar",
        ]:
            series = country_data[country_name][variable_name]
            series = series.reindex(snapshots)
            series = series.interpolate().ffill().bfill()
            country_data[country_name][variable_name] = series

    line_capacities_mw = {}
    for bus0, bus1 in NETWORK_EDGES:
        line_capacities_mw[(bus0, bus1)] = get_observed_line_capacity_mw(bus0, bus1)

    return country_data, line_capacities_mw, snapshots


if __name__ == "__main__":
    country_data, line_capacities_mw, snapshots = build_country_data_and_lines()

    print("Installed capacities in 2024 (MW):")
    for country_name in country_data:
        print(f"\n{country_name}")
        print(pd.Series(country_data[country_name]["capacities_mw"]))

    print("\nInterconnector capacities used in the model (MW):")
    for edge, value in line_capacities_mw.items():
        print(f"{edge}: {value}")
