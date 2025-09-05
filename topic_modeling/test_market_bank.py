#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 16:01:07 2025

@author: cristiantobar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import os

    
def load_timeseries(csv_path, col_cat):

    col_val="Value"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
        
    # Detectar columna de tiempo
    col_time = 'TIME' if 'TIME' in df.columns else 'Time'
    # Detectar columna de categoría (la que no sea TIME, Value ni Region/Frequency)
    #cat_candidates = [c for c in df.columns if c not in [col_time, col_val, 'Region', 'FREQUENCY','Frequency']]
    #col_cat = cat_candidates[0]
    
    # Parsear fechas (anual, trimestral o mensual)
    def parse_date(x):
        s = str(x)
        # YYYY-MM
        try: return pd.to_datetime(s, format="%Y-%m")
        except: pass
        # YYYYQn
        m = re.match(r"^(\d{4})Q([1-4])$", s)
        if m: 
            return pd.Period(year=int(m.group(1)), quarter=int(m.group(2))).to_timestamp()
        # YYYY
        m = re.match(r"^(\d{4})$", s)
        if m:
            return pd.Timestamp(year=int(m.group(1)), month=1, day=1)
        return pd.to_datetime(s, errors="coerce")

    df['date'] = df[col_time].map(parse_date)
    df[col_val] = pd.to_numeric(df[col_val], errors='coerce')

    df_long = df[['date', col_cat, col_val]].rename(columns={col_cat:'category', col_val:'value'})
    df_wide = df_long.pivot(index='date', columns='category', values='value').sort_index()
    return df_wide


# ===== Uso =====

folder_path = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/base_datos_banco_belgica/datos_nbb_stats"

manifest = [
    {"label": "mark_1",  "filename": "1_producer price indices.csv",                "cat_col": "Product - 2021=100",                    "variables": ['Manufacture of concrete products for construction','Manufacture of construction products, in baked clay','Manufacture of metal products for construction','Manufacture of plaster products for construction'],   "frequency":"M"},
    {"label": "mark_2",  "filename": "2_balance of payments.csv",                   "cat_col": "Item",                                  "variables": ['Construction services'],                                                                                                                                                                                         "frequency":"M"},
    {"label": "mark_3",  "filename": "3_monthly business surveys.csv",              "cat_col": "Sector",                                "variables": ['Construction installation','Construction of residential buildings'],                                                                                                                                             "frequency":"M"},
    {"label": "mark_4a", "filename": "4_Annual accounts of companies_1.csv",        "cat_col": "Grouping of activity sectors",          "variables": ['Construction of residential and non-residential buildings; civil engineering','General construction of buildings and civil engineering works','Wholesale of wood, paint, varnish and construction materials'],   "frequency":"A"},
    {"label": "mark_4b", "filename": "4_Annual accounts of companies_2.csv",        "cat_col": "Grouping of activity sectors",          "variables": ['Construction of residential and non-residential buildings; civil engineering','General construction of buildings and civil engineering works','Wholesale of wood, paint, varnish and construction materials'],   "frequency":"A"},
    {"label": "mark_4c", "filename": "4_Annual accounts of companies_3.csv",        "cat_col": "Grouping of activity sectors",          "variables": ['Construction of residential and non-residential buildings; civil engineering','General construction of buildings and civil engineering works','Wholesale of wood, paint, varnish and construction materials'],   "frequency":"A"},
    {"label": "mark_4d", "filename": "4_Annual accounts of companies_4.csv",        "cat_col": "Grouping of activity sectors",          "variables": ['Construction of residential and non-residential buildings; civil engineering','General construction of buildings and civil engineering works','Wholesale of wood, paint, varnish and construction materials'],   "frequency":"A"},
    {"label": "mark_5",  "filename": "5_Financial ratios of companies.csv",         "cat_col": "Grouping of activity sectors",          "variables": ['Construction of residential and non-residential buildings; civil engineering','General construction of buildings and civil engineering works','Wholesale of wood, paint, varnish and construction materials'],   "frequency":"A"},
    {"label": "mark_6",  "filename": "6_All social balance sheets.csv",             "cat_col": "Grouping of activity sectors",          "variables": ['Construction of residential and non-residential buildings; civil engineering','General construction of buildings and civil engineering works','Wholesale of wood, paint, varnish and construction materials'],   "frequency":"A"},
    {"label": "mark_7",  "filename": "7_Other economic indicators_construction.csv","cat_col": "Building permits / Buildings started",  "variables": ['Building permits concerning month of concession'],                                                                                                                                                               "frequency":"M"},
    {"label": "mark_8",  "filename": "8_employment.csv",                            "cat_col": "Indicator",                             "variables": ['Employment (thousands of persons)'],                                                                                                                                                                             "frequency":"A"},
    {"label": "mark_9",  "filename": "9_foreign_trade_national.csv",                "cat_col": "PRODUCT",                               "variables": "*",                                                                                                                                                                                                               "frequency":"M"},
    {"label": "mark_10", "filename": "10_industrial_production.csv",                "cat_col": "Sector",                                "variables": ['Construction of buildings, development of building projects'],                                                                                                                                                   "frequency":"M"},
    #{"label": "mark_11", "filename": "11_major_components_branch.csv",              "cat_col": "Industry breakdown",                    "variables": ['FF Construction'],                                                                                                                                                                                               "frequency":"A"},
    {"label": "mark_12", "filename": "12_unemployed_job_seekers.csv",               "cat_col": "Activity",                              "variables": ['Construction'],                                                                                                                                                                                                  "frequency":"M"},
    {"label": "mark_13", "filename": "13_gov_spending.csv",                         "cat_col": "Government function",                   "variables": ['04.4 Mining, manufacturing and construction'],                                                                                                                                                                   "frequency":"A"},
    #{"label": "mark_14", "filename": "14_regional_accounts_64.csv",                 "cat_col": "Branch of industry",                    "variables": ['Construction (41-43)'],                                                                                                                                                                                          "frequency":"A"},
    {"label": "mark_15", "filename": "15_supply_and_use_table.csv",                 "cat_col": "Products",                              "variables": ['41-43 - Constructions and construction works'],                                                                                                                                                                  "frequency":"A"},
]

datasets = {}
datasets_net = {}
for entry in manifest:
    path = folder_path + "/" + entry["filename"]
    datasets[entry["label"]] = load_timeseries(path, entry["cat_col"])
    df_neto = datasets[entry["label"]]
    
    vars_spec = entry.get("variables")

    if vars_spec in (None, "", "*"):
        # 1) Todas las columnas
        df_sel = df_neto.copy()

    else:
        # 3) Lista de columnas concretas (con manejo de ausentes)
        cols = [c for c in vars_spec if c in df_neto.columns]
        missing = sorted(set(vars_spec) - set(cols))
        if missing:
            print(f"[{entry['label']}] Aviso: columnas no encontradas y omitidas: {missing}")
        df_sel = df_neto[cols]

    datasets_net[entry["label"]] = df_sel

# 1. Concatenar los cuatro
df_concat = pd.concat(
    [datasets_net[k] for k in ["mark_4a", "mark_4b", "mark_4c", "mark_4d"]],
    axis=0
).sort_index()

# 2. Guardarlo con un nuevo label
datasets_net["mark_4"] = df_concat

# 3. Eliminar los anteriores
for k in ["mark_4a", "mark_4b", "mark_4c", "mark_4d"]:
    datasets_net.pop(k, None)

import pandas as pd
import numpy as np

# ======================= utilidades =======================
def _target_index(freq: str, start_year=2011, end_year=2023) -> pd.DatetimeIndex:
    freq = freq.upper()
    if freq == "A":
        return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="YS")
    elif freq == "M":
        return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")
    else:
        raise ValueError(f"Frecuencia no soportada: {freq} (usa 'A' o 'M')")

def _avg_rate(series: pd.Series, k: int = 12) -> float | None:
    s = series.dropna()
    if len(s) < 3:
        return None
    tail = s.iloc[-(k+1):] if len(s) > k+1 else s
    pct = tail.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if pct.empty:
        return None
    return float(pct.mean())

def _avg_slope(series: pd.Series, k: int = 12) -> float | None:
    s = series.dropna()
    if len(s) < 3:
        return None
    tail = s.iloc[-(k+1):] if len(s) > k+1 else s
    dif = tail.diff().dropna()
    if dif.empty:
        return None
    return float(dif.mean())

def _cagr(series: pd.Series, years_window: int = 5) -> float | None:
    s = series.dropna()
    if len(s) < 2:
        return None
    tail = s.iloc[-(years_window+1):] if len(s) > years_window+1 else s
    n = len(tail) - 1
    if n < 1:
        return None
    first, last = tail.iloc[0], tail.iloc[-1]
    if first <= 0 or last <= 0:
        return None
    try:
        return (last / first) ** (1 / n) - 1
    except Exception:
        return None

def _linear_slope(series: pd.Series, years_window: int = 5) -> float | None:
    s = series.dropna()
    if len(s) < 2:
        return None
    tail = s.iloc[-(years_window+1):] if len(s) > years_window+1 else s
    dif = tail.diff().dropna()
    if dif.empty:
        return None
    return float(dif.mean())

def _normalize_index_to_period_start(s: pd.Series, freq: str) -> pd.Series:
    freq = freq.upper()
    s2 = s.copy()
    if freq == "A":
        s2.index = pd.to_datetime(s2.index).to_period("Y").to_timestamp(how="start")
    elif freq == "M":
        s2.index = pd.to_datetime(s2.index).to_period("M").to_timestamp(how="start")
    else:
        raise ValueError(f"Frecuencia no soportada: {freq}")
    return s2

def build_freq_map(manifest):
    freq = {}
    for e in manifest:
        lbl = e["label"]
        f = e["frequency"].upper()
        freq[lbl] = f
        # alias: label base (quita sufijo a/b/c/d si existe)
        base = re.sub(r'[a-d]$', '', lbl)
        if base and base not in freq:
            freq[base] = f
    return freq

# =============== completa UNA serie según manifest ===============
def _complete_one_series_with_freq(s: pd.Series, freq: str,
                                   start_year=2011, end_year=2023) -> tuple[pd.Series, pd.Series]:
    """
    Devuelve (serie_completada, status) con status en {'observed','interp','proj_fwd','proj_back','fill'}.
    Usa la frecuencia dada por manifest ('A' o 'M').
    """
    freq = freq.upper()
    s_norm = _normalize_index_to_period_start(s, freq)
    target_idx = _target_index(freq, start_year, end_year)

    # s2 = s_norm.sort_index()
    # observed_mask = pd.Series(False, index=target_idx)
    # if s2.index.size:
    #     observed_mask.loc[s2.index.intersection(target_idx)] = True

    # # Reindex a ventana objetivo
    # s2 = s2.reindex(target_idx)

    # # Interpolación interna
    # s_interp = s2.copy()
    # s_interp_intermediate = s_interp.interpolate(method="linear", limit_area="inside")

    # status = pd.Series(index=target_idx, dtype="object")
    # status[:] = None
    # status[observed_mask] = "observed"

    # internal_gaps = s2.isna() & s_interp_intermediate.notna()
    # status[internal_gaps & status.isna()] = "interp"
    # s_filled = s_interp_intermediate.copy()
    
    # 2) Observados verdaderos = índices con valor NO NaN antes de reindexar
    observed_idx = s_norm.dropna().index
    observed_mask = pd.Series(False, index=target_idx)
    observed_mask.loc[observed_idx.intersection(target_idx)] = True
    
    # 3) Reindex al rango objetivo (introduce NaN en huecos)
    pre = s_norm.reindex(target_idx)
    
    # 4) Interpolación SOLO de huecos internos
    post = pre.interpolate(method="linear", limit_area="inside")
    
    # 5) Interp_mask = lugares que eran NaN y ahora tienen valor
    interp_mask = pre.isna() & post.notna()
    
    # 6) Inicializa status y etiqueta sin pisar nada después
    status = pd.Series(index=target_idx, dtype="object")
    status[:] = None
    status[observed_mask] = "observed"
    status[interp_mask & status.isna()] = "interp"
    
    # 7) Continúa con proyección usando `post` como base:
    s_filled = post.copy()

    # ---------- Proyección hacia adelante ----------
    tail_nans = s_filled.isna() & (
        s_filled.index > s_filled.first_valid_index() if s_filled.first_valid_index() is not None else False
    )
    if tail_nans.any():
        last_obs_idx = s_filled.last_valid_index()
        if last_obs_idx is not None:
            last_obs_pos = s_filled.index.get_loc(last_obs_idx)
            if freq == "M":
                rate = _avg_rate(s_filled[:last_obs_idx], k=12)
                slope = _avg_slope(s_filled[:last_obs_idx], k=12)
                use_rate = rate is not None and (s_filled[:last_obs_idx].dropna() > 0).all()
                for i in range(last_obs_pos + 1, len(s_filled)):
                    prev = s_filled.iat[i - 1]
                    if np.isnan(prev):
                        break
                    if use_rate:
                        s_filled.iat[i] = prev * (1 + rate)
                    elif slope is not None:
                        s_filled.iat[i] = prev + slope
                    else:
                        s_filled.iat[i] = prev  # ffill plano
                status[(status.isna()) & s_filled.notna()] = "proj_fwd"
            elif freq == "A":
                r = _cagr(s_filled[:last_obs_idx], years_window=5)
                m = None if r is not None else _linear_slope(s_filled[:last_obs_idx], years_window=5)
                for i in range(last_obs_pos + 1, len(s_filled)):
                    prev = s_filled.iat[i - 1]
                    if np.isnan(prev):
                        break
                    if r is not None:
                        s_filled.iat[i] = prev * (1 + r)
                    elif m is not None:
                        s_filled.iat[i] = prev + m
                    else:
                        s_filled.iat[i] = prev
                status[(status.isna()) & s_filled.notna()] = "proj_fwd"

    # ---------- Proyección hacia atrás ----------
    head_nans = s_filled.isna() & (
        s_filled.index < s_filled.last_valid_index() if s_filled.last_valid_index() is not None else False
    )
    if head_nans.any():
        first_obs_idx = s_filled.first_valid_index()
        if first_obs_idx is not None:
            first_obs_pos = s_filled.index.get_loc(first_obs_idx)
            if freq == "M":
                s_tmp = s_filled[first_obs_idx:].copy()
                k = max(1, min(12, s_tmp.notna().sum() - 1))
                rate = _avg_rate(s_tmp, k=k)
                slope = _avg_slope(s_tmp, k=k)
                use_rate = rate is not None and (s_tmp.dropna() > 0).all()
                for i in range(first_obs_pos - 1, -1, -1):
                    nxt = s_filled.iat[i + 1]
                    if np.isnan(nxt):
                        break
                    if use_rate:
                        s_filled.iat[i] = nxt / (1 + rate)
                    elif slope is not None:
                        s_filled.iat[i] = nxt - slope
                    else:
                        s_filled.iat[i] = nxt
                status[(status.isna()) & s_filled.notna()] = "proj_back"
            elif freq == "A":
                s_tmp = s_filled[first_obs_idx:].copy()
                k = max(1, min(5, s_tmp.notna().sum() - 1))
                r = _cagr(s_tmp, years_window=k)
                m = None if r is not None else _linear_slope(s_tmp, years_window=k)
                for i in range(first_obs_pos - 1, -1, -1):
                    nxt = s_filled.iat[i + 1]
                    if np.isnan(nxt):
                        break
                    if r is not None:
                        s_filled.iat[i] = nxt / (1 + r)
                    elif m is not None:
                        s_filled.iat[i] = nxt - m
                    else:
                        s_filled.iat[i] = nxt
                status[(status.isna()) & s_filled.notna()] = "proj_back"

    # Último recurso: ffill/bfill
    if s_filled.isna().any():
        before = s_filled.copy()
        s_filled = s_filled.ffill().bfill()
        newly = before.isna() & s_filled.notna()
        status[newly & status.isna()] = "fill"

    status[observed_mask] = "observed"
    return s_filled, status

# =============== pipeline principal con manifest ===============
def complete_timeseries_with_manifest(manifest: list[dict],
                                      datasets_net: dict[str, pd.DataFrame],
                                      start_year: int = 2011, end_year: int = 2023):
    """
    Usa 'frequency' del manifest por label para completar 2011–2023.
    Retorna:
      - completed: dict[label] -> DataFrame completo
      - status:    dict[label] -> DataFrame con etiquetas ('observed','interp','proj_fwd','proj_back','fill')
    """
    # mapa label -> frequency
    # freq_map = {entry["label"]: entry["frequency"].upper() for entry in manifest}
    freq_map = build_freq_map(manifest)

    completed = {}
    status_maps = {}

    for label, df in datasets_net.items():
        if df is None or df.empty:
            completed[label] = df
            status_maps[label] = df
            continue
        
        # if label == "mark_9":
        #     breakpoint()
        
        if label not in freq_map:
            raise KeyError(f"No se encontró 'frequency' para '{label}' en el manifest")

        freq = freq_map[label]
        # normalizar índice de todo el DataFrame a inicio de periodo
        if freq == "A":
            idx_norm = pd.to_datetime(df.index).to_period("Y").to_timestamp(how="start")
        elif freq == "M":
            idx_norm = pd.to_datetime(df.index).to_period("M").to_timestamp(how="start")
        else:
            raise ValueError(f"Frecuencia no soportada: {freq}")

        df_norm = df.copy()
        df_norm.index = idx_norm

        cols_done = []
        stat_done = []
        for col in df_norm.columns:
            s_comp, s_stat = _complete_one_series_with_freq(df_norm[col], freq=freq,
                                                            start_year=start_year, end_year=end_year)
            s_comp.name = col
            s_stat.name = col
            cols_done.append(s_comp)
            stat_done.append(s_stat)

        df_comp = pd.concat(cols_done, axis=1)
        df_stat = pd.concat(stat_done, axis=1)

        completed[label] = df_comp
        status_maps[label] = df_stat

    return completed, status_maps

# ============ USO ============
completed, status = complete_timeseries_with_manifest(manifest, datasets_net, start_year=2011, end_year=2023)
# Ejemplo:
# completed['mark_1'].head()
# status['mark_1'].head()

# ------------------ resumen por dominio ------------------
def quality_summary(status_maps: dict) -> pd.DataFrame:
    """
    Calcula, por dominio (label), el % de celdas con cada estatus:
    observed / interp / proj_fwd / proj_back / fill, y % imputado (no-observed).
    """
    statuses = ["observed", "interp", "proj_fwd", "proj_back", "fill"]
    rows = []

    for label, st in status_maps.items():
        if st is None or len(st) == 0:
            continue

        ser = st.stack()                          # Serie larga: una etiqueta por celda
        total = ser.notna().sum()                 # total de celdas con etiqueta
        counts = ser.value_counts()               # conteo por estatus

        row = {"domain": label, "total_cells": int(total)}
        for s in statuses:
            c = int(counts.get(s, 0))
            row[f"{s}_count"] = c
            row[f"{s}_pct"] = (c / total * 100.0) if total > 0 else np.nan

        imputed = total - int(counts.get("observed", 0))
        row["imputed_pct"] = (imputed / total * 100.0) if total > 0 else np.nan

        rows.append(row)

    summary = pd.DataFrame(rows).set_index("domain")
    return summary

# ------------------ construir orden y nombres desde manifest ------------------
def build_order_and_names(manifest: list[dict], summary_index) -> tuple[list, dict]:
    """
    Devuelve:
      - desired_order: lista de labels en el orden del manifest (con soporte para mark_4 combinado)
      - label_to_name: mapeo label -> filename (sin .csv); para combinados agrega ' (combined)'
    """
    # Orden base: el orden del manifest
    manifest_labels = [e["label"] for e in manifest]
    desired_order = manifest_labels.copy()

    # Si existe un mark_4 combinado en summary, insértalo después de mark_3 si no está ya
    if "mark_4" in summary_index and "mark_4" not in desired_order:
        # buscar dónde insertar: después de mark_3 si existe, si no al inicio
        try:
            pos = desired_order.index("mark_3") + 1
        except ValueError:
            pos = 0
        desired_order.insert(pos, "mark_4")

    # Filtrar a solo los labels que realmente existen en summary
    desired_order = [lbl for lbl in desired_order if lbl in summary_index]

    # Mapear label -> filename (sin .csv)
    label_to_name = {e["label"]: e["filename"].replace(".csv", "") for e in manifest}

    # Soporte para mark_4 combinado (si está en summary pero no en manifest)
    if "mark_4" in summary_index and "mark_4" not in label_to_name:
        # busca cualquier entrada mark_4[a-d] en el manifest para tomar el nombre base
        candidates = [e for e in manifest if re.match(r"mark_4[a-d]$", e["label"])]
        if candidates:
            base_name = candidates[0]["filename"].replace(".csv", "")
            label_to_name["mark_4"] = base_name + " (combined)"
        else:
            label_to_name["mark_4"] = "mark_4 (combined)"

    return desired_order, label_to_name

# ------------------ generar tabla y gráfico con nombres descriptivos ------------------
def quality_report(status_maps: dict, manifest: list[dict]):
    summary = quality_summary(status_maps)

    desired_order, label_to_name = build_order_and_names(manifest, summary.index)

    # Reordenar summary
    summary = summary.reindex(desired_order)

    # Tabla: solo columnas de porcentaje
    cols_pct = [c for c in summary.columns if c.endswith("_pct")]
    tabla_calidad = summary[cols_pct].round(2)

    # Renombrar índices a nombres de archivo
    tabla_calidad_named = tabla_calidad.rename(index=label_to_name)

    print("\n=== Tabla de calidad por dominio (% de celdas) ===")
    print(tabla_calidad_named)

    # Gráfico de barras apiladas
    plot_cols = [f"{s}_pct" for s in ["observed", "interp", "proj_fwd", "proj_back", "fill"]]
    plot_df = summary[plot_cols].fillna(0)
    plot_df_named = plot_df.rename(index=label_to_name)

    ax = plot_df_named.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_ylabel("% de celdas")
    ax.set_xlabel("Fuente de datos")
    ax.set_ylim(0, 100)
    ax.set_title("Calidad de datos por dominio (observado vs imputado)")
    ax.legend(title="Estatus", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return tabla_calidad_named

# =================== USO ===================
tabla_calidad_named = quality_report(status, manifest)

#excluded = {"mark_9", "mark_11", "mark_14",}  # o la lista de los que no pasan el filtro

excluded = {"mark_9"}

completed_net = {k: v for k, v in completed.items() if k not in excluded}
status_net    = {k: v for k, v in status.items() if k not in excluded}

# Ruta de la carpeta donde están los archivos Excel
carpeta_excel = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/datos_schedules_construccion"

# Ruta del nuevo archivo Excel que quieres agregar
ruta_apus_traduccion = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/apus_traduccion.xlsx"

# Ruta del archivo excel consolidado de los papers
ruta_consolidado_proj = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/DSLIB_Analysis_Scheet.xlsx"

output_folder_2 = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/procesamiento_lenguaje_natural/petri_net_modular"

keep_outliers = False
var_pred = ['Duration', 'Cost'] 
pred_choose = var_pred[0]

# Obtener la lista de archivos en la carpeta
archivos_excel = [f for f in os.listdir(carpeta_excel) if f.endswith(".xlsx") or f.endswith(".xls")]

# Diccionario para almacenar los DataFrames de cada archivo
hojas_excel = {}

# Recorrer los archivos y leer la primera hoja
for archivo in archivos_excel:
    
# archivo = 'C2016-38 Passive house construction.xlsx'
    
# if archivo == 'C2016-38 Passive house construction.xlsx':
    
    ruta_completa = os.path.join(carpeta_excel, archivo)
        
    try:
        # Obtener todas las hojas del archivo
        xls = pd.ExcelFile(ruta_completa)
        hojas = xls.sheet_names
        
        # Inicializar diccionario para este archivo
        hojas_excel[archivo] = {}

        # Leer la primera hoja
        primera_hoja = pd.read_excel(ruta_completa, sheet_name=0)
        primera_hoja.columns = primera_hoja.iloc[0]  # Asignar la primera fila como encabezado
        primera_hoja = primera_hoja[1:].reset_index(drop=True)
        hojas_excel[archivo]["Primera_Hoja"] = primera_hoja

        # Leer la hoja "AGENDA" si existe
        if "Agenda" in hojas:
            df_agenda = pd.read_excel(xls, sheet_name="Agenda", header=None)
            #agenda.columns = agenda.iloc[0]  # Asignar encabezado
            #agenda = agenda[1:].reset_index(drop=True)

            # Renombrar columnas esperadas
            df_agenda.columns = ["Working Hours", "Status", "no data1","Working Days", "Working Days Status", "no data2", "Holidays"]
            df_agenda = df_agenda[1:].reset_index(drop=True)  # Elimina las tres primeras filas y reinicia el índice
            df_agenda.dropna(how="all", inplace=True)  # Eliminar filas vacías
            
            # Eliminar columnas innecesarias
            df_agenda.drop(columns=["no data1", "no data2"], errors="ignore", inplace=True)

            # Eliminar la última fila del DataFrame
            #df_agenda = df_agenda.iloc[:-1]            

            if "Holidays" in df_agenda.columns:
                # Convertir a string, limpiar valores nulos y contar las celdas no vacías
                holidays_count = int(df_agenda["Holidays"].astype(str).str.strip()
                                     .replace(["nan", "NaT", "None", "<NA>", "NaN"], "")
                                     .ne("").sum())

            # Calcular métricas
            working_hours_per_day = df_agenda[df_agenda["Status"] == "Yes"].shape[0]
            working_days_count = df_agenda[df_agenda["Working Days Status"] == "Yes"].shape[0]
        
        tp = None
        
        # Filtrar hojas que empiezan con "TP" seguido de un número
        hojas_tp = [h for h in hojas if h.startswith("TP") and h[2:].isdigit()]

        # Si hay hojas "TP", seleccionar la de mayor número
        if hojas_tp:
            hoja_tp_mayor = max(hojas_tp, key=lambda h: int(h[2:]))  # Obtener la hoja con el número más grande
            tp = pd.read_excel(xls, sheet_name=hoja_tp_mayor, header=None)
                      
            # Usar la segunda fila como nombres de columna y eliminar las tres primeras filas correctamente
            tp.columns = tp.iloc[3]  # Toma la tercera fila como encabezado (índice 2)
            tp = tp[4:].reset_index(drop=True)  # Elimina las tres primeras filas y reinicia el índice
            hojas_excel[archivo]["TP_Max"] = tp
            
        # Añadir las métricas de "AGENDA" a todas las filas de "Primera_Hoja"
            primera_hoja["Working Hours Per Day"] = working_hours_per_day
            primera_hoja["Working Days Count"] = working_days_count
            primera_hoja["Holidays Count"] = holidays_count
        print(f"✅ Leído correctamente: {archivo}")
              
        
    # Intentar fusionar solo si "TP_Max" existe y tiene la columna en común con "Primera_Hoja"
        columna_comun = "ID"  # Ajusta esto con el nombre real de la columna que comparten
        
        if tp is not None and columna_comun in primera_hoja.columns and columna_comun in tp.columns:
            fusionado = pd.merge(primera_hoja, tp, on=columna_comun, how="left")
            
            # Eliminar columnas que tienen 'nan' como encabezado
            fusionado = fusionado.loc[:, ~fusionado.columns.isna()]

            # También podrías eliminar columnas con nombres vacíos o espacios en blanco por seguridad
            fusionado = fusionado.loc[:, fusionado.columns.str.strip().astype(bool)]

            hojas_excel[archivo]["Fusionado"] = fusionado
            print(f"✅ Archivo {archivo} fusionado correctamente.")
        else:
            hojas_excel[archivo]["Fusionado"] = primera_hoja  # Si no hay "TP_Max", guardamos solo "Primera_Hoja"
            print(f"⚠ No se encontró la hoja 'TP' o la columna '{columna_comun}' en {archivo}. Se guarda solo 'Primera_Hoja'.")

    except Exception as e:
        breakpoint()
        print(f"❌ Error al leer {archivo}: {e}")

# ✅ **Agregar el nuevo archivo `apus_traduccion.xlsx` al diccionario**
try:
    df_apus_traduccion = pd.read_excel(ruta_apus_traduccion)  # Carga todas las hojas
    hojas_excel["apus_traduccion.xlsx"] = df_apus_traduccion  # Agregarlo al diccionario
    print("✅ Archivo 'apus_traduccion.xlsx' cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar 'apus_traduccion.xlsx': {e}")

try:
    df_consolidado_proj = pd.read_excel(ruta_consolidado_proj, sheet_name='all_data_combining')  # Carga todas las hojas 
    print("✅ Archivo 'df_consolidado_proj.xlsx' cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar 'df_consolidado_proj.xlsx': {e}")

# Obtener todos los DataFrames fusionados
df_fusionados = [(archivo, hojas_excel[archivo]["Fusionado"]) for archivo in hojas_excel if "Fusionado" in hojas_excel[archivo]]

# Contar la frecuencia de cada conjunto de columnas
from collections import Counter

columnas_contador = Counter(tuple(df.columns) for _, df in df_fusionados)
columnas_mayoritarias = max(columnas_contador, key=columnas_contador.get)  # Obtener las columnas más comunes (47)

# Asegurar que todos los DataFrames tengan estas columnas
df_homogeneos = []

for idx, (archivo, df) in enumerate(df_fusionados, start=1):  # ID de archivo comienza desde 1
    df = df.reindex(columns=columnas_mayoritarias)  # Añadir columnas faltantes con NaN
    df.insert(0, "Project_ID", idx)
    df.insert(1, "Filename", archivo)
    df_homogeneos.append(df)

# Concatenar todos los DataFrames en un dataset maestro
dataset_maestro = pd.concat(df_homogeneos, ignore_index=True)

# Extraer el código del proyecto de 'Filename' en dataset_maestro
dataset_maestro["Code"] = dataset_maestro["Filename"].str.extract(r"^(C\d{4}-\d{2})")

# Realizar la fusión con df_consolidado_proj
dataset_maestro_fusionado = dataset_maestro.merge(df_consolidado_proj, on="Code", how="left")

# Filtrar las filas que serán eliminadas (aquellas donde ambas columnas sean NaN)
filas_eliminadas = dataset_maestro_fusionado[dataset_maestro_fusionado[['Predecessors', 'Successors']].isna().all(axis=1)]
filas_eliminadas = filas_eliminadas.reset_index(drop=True)

# Crear el dataset limpio eliminando esas filas
dataset_maestro_fusionado_limpio = dataset_maestro_fusionado.dropna(subset=['Predecessors', 'Successors'], how='all').reset_index(drop=True)

dataset_maestro_fusionado_limpio = dataset_maestro_fusionado_limpio.drop(columns = [
                                                                               'Baseline Start_x', 
                                                                               'Baseline End_x',
                                                                               'Baseline duration (in calendar days)',
                                                                               'Name_y',
                                                                               'Baseline Start_y',
                                                                               'Baseline End_y',
                                                                               'Duration_y',
                                                                               'Resource Demand_y', 
                                                                               'Resource Cost_y',
                                                                               'Fixed Cost_y', 
                                                                               'Cost/Hour_y', 
                                                                               'Variable Cost_y', 
                                                                               'Total Cost_y',
                                                                               'PRC', 
                                                                               'Remaining Duration',
                                                                               'PRC Dev',
                                                                               'Remaining Cost',
                                                                               'Percentage Completed',
                                                                               'Tracking',
                                                                               'Relative baseline duration',
                                                                               'Percentage completed',
                                                                               'Relative baseline cost',
                                                                               'Percentage completed', 
                                                                               'Code',
                                                                               'Project name', 
                                                                               'Sector', 
                                                                               'Keywords',
                                                                               'Duration', 
                                                                               'Cost'])



df = dataset_maestro_fusionado_limpio

def _label_to_filename_map(manifest):
    """label -> filename (sin .csv)"""
    return {e["label"]: e["filename"].replace(".csv","") for e in manifest}

def _build_freq_map_with_alias(manifest):
    """label -> frequency (y alias base para colapsados, p.ej. mark_4)."""
    freq = {}
    for e in manifest:
        lbl = e["label"]
        f = e["frequency"].upper()
        freq[lbl] = f
        base = re.sub(r'[a-d]$', '', lbl)
        if base and base not in freq:
            freq[base] = f
    return freq

def _prepare_time_anchors(df: pd.DataFrame, actual_start_col: str):
    """Agrega columnas ancla para merge temporal (sin modificar si no existe Actual Start)."""
    if actual_start_col not in df.columns:
        raise KeyError(f"No existe la columna '{actual_start_col}' en el DataFrame de actividades.")
    df = df.copy()
    # convertir a datetime (valores inválidos -> NaT)
    df["_nbb_actual_start_dt"] = pd.to_datetime(df[actual_start_col], errors="coerce")
    # anclas mensual/anual (NaT si no hay fecha)
    df["_nbb_anchor_M"] = df["_nbb_actual_start_dt"].dt.to_period("M").dt.to_timestamp(how="start")
    df["_nbb_anchor_A"] = df["_nbb_actual_start_dt"].dt.to_period("Y").dt.to_timestamp(how="start")
    return df

def _merge_one_domain(df_act: pd.DataFrame,
                      nbb_df: pd.DataFrame,
                      freq: str,
                      prefix: str,
                      on_out_of_range: str = "nan"):
    """
    Enrich por un dominio (label). 
    freq: 'M' o 'A'. policy: 'nan' (default) o 'nearest'/'ffill'/'bfill' si quisieras cambiar luego.
    """
    if nbb_df is None or nbb_df.empty:
        # crea columnas vacías con NaN
        for col in nbb_df.columns if nbb_df is not None else []:
            df_act[f"{prefix}{col}"] = np.nan
        return df_act

    # Prepara índice para merge exacto
    idx_col = "_nbb_index_key"
    tmp = nbb_df.copy()
    tmp = tmp.sort_index()
    tmp[idx_col] = tmp.index

    # Política de fuera-de-rango (por ahora 'nan' => no alteramos tmp)
    # Si quisieras nearest, podríamos reindexar a un rango y hacer método='nearest'

    # Elegir ancla
    anchor_col = "_nbb_anchor_M" if freq == "M" else "_nbb_anchor_A"

    # Hacemos merge left por fecha exacta
    right = tmp.set_index(idx_col).reset_index()
    # Prefijar nombres para evitar colisiones
    right = right.rename(columns={c: f"{prefix}{c}" for c in nbb_df.columns})
    # importante: mantener la columna clave para el merge
    right = right.rename(columns={idx_col: anchor_col})

    # Merge
    out = df_act.merge(right, on=anchor_col, how="left")

    # Si quisieras una bandera de “fuera de rango o sin fecha”, puedes crearla:
    # out[f"{prefix}__missing"] = out[anchor_col].isna()

    return out

def add_nbb_features_to_activities(df_activities: pd.DataFrame,
                                   completed_net: dict,
                                   manifest: list,
                                   actual_start_col: str = "Actual Start",
                                   use_filenames_as_prefix: bool = False,
                                   on_out_of_range: str = "nan",
                                   keep_anchor_cols: bool = False) -> pd.DataFrame:
    """
    Enriquecer actividades con NBB.stat de completed_net según manifest.
    Política:
      - Si Actual Start es NaT -> columnas agregadas quedan NaN.
      - Si la fecha no existe en NBB (fuera de 2011–2023) -> NaN.
    """
    df = _prepare_time_anchors(df_activities, actual_start_col)

    # mapas auxiliares
    freq_map = _build_freq_map_with_alias(manifest)
    label2name = _label_to_filename_map(manifest)

    # Iterar por dominios presentes en completed_net
    out = df
    for label, nbb_df in completed_net.items():
        freq = freq_map.get(label)
        if freq is None:
            # si falta freq, omitir con aviso suave
            # print(f"[WARN] No hay frequency para {label}; se omite.")
            continue
        freq = freq.upper()
        if freq not in ("A","M"):
            # print(f"[WARN] Frecuencia no soportada {freq} para {label}; se omite.")
            continue

        # El prefijo puede ser por label o por filename
        base = label2name.get(label, label)
        prefix = f"nbb_{base}__"

        # Merge del dominio
        out = _merge_one_domain(out, nbb_df, freq=freq, prefix=prefix, on_out_of_range=on_out_of_range)

    # Limpiar columnas auxiliares
    if not keep_anchor_cols:
        out = out.drop(columns=[c for c in ["_nbb_actual_start_dt","_nbb_anchor_M","_nbb_anchor_A"] if c in out.columns])

    return out

# ================== USO ==================
# IMPORTANTE: asegúrate de NO eliminar 'Actual Start' antes de este paso

def build_dataset():
    df_actividades_enriquecido = add_nbb_features_to_activities(
        df_activities=dataset_maestro_fusionado_limpio,   # tu DF con 'Actual Start'
        completed_net=completed_net,                      # dict label -> DF 2011-2023
        manifest=manifest,                                # tu manifest con 'frequency'
        actual_start_col="Actual Start",
        use_filenames_as_prefix=True,                     # True si quieres prefijo por filename del manifest
        on_out_of_range="nan",                            # política: 'nan' (default)
        keep_anchor_cols=False                            # True si quieres ver las columnas ancla
    )
    
    return df_actividades_enriquecido, completed_net
# df = df_actividades_enriquecido