#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 13:12:05 2025

@author: cristiantobar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Literal


file_path_db_market = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/base_datos_3/datos/database_2.xlsx"

# Cargar la hoja principal
df = pd.read_excel(file_path_db_market, sheet_name="Hoja1")

# Mostrar las primeras filas y columnas para inspección
df.head(), df.columns

var_mapping = {
    "V_1": "Project locality (zip codes)",
    "V_2": "Total floor area (m2)",
    "V_3": "Lot area (m2)",
    "V_4": "Preliminary construction cost (beginning, 10^7 IRR)",
    "V_5": "Preliminary construction cost (beginning, 10^4 IRR)",
    "V_6": "Equivalent preliminary cost (base year, 10^4 IRR)",
    "V_7": "Duration of construction (time units)",
    "V_8": "Price/unit at beginning (10^4 IRR)",
    "V-9": "Actual sales prices (10^4 IRR) [output]",
    "V-10": "Actual construction costs (10^4 IRR) [output]",
    "V_11": "Number of building permits issued",
    "V_12": "Building services index (BSI, base year)",
    "V_13": "Wholesale price index (WPI, base year)",
    "V_14": "Total floor area of building permits (m2)",
    "V_15": "Cumulative liquidity (10^7 IRR)",
    "V_16": "Private sector investment in new buildings (10^7 IRR)",
    "V_17": "Land price index (base year, 10^7 IRR)",
    "V_18": "Number of loans extended by banks",
    "V_19": "Amount of loans extended (10^7 IRR)",
    "V_20": "Loan interest rate (%)",
    "V_21": "Avg construction cost (completion, 10^4 IRR/m2)",
    "V_22": "Avg construction cost (beginning, 10^4 IRR/m2)",
    "V_23": "Official exchange rate (IRR/USD)",
    "V_24": "Unofficial exchange rate (IRR/USD)",
    "V_25": "Consumer price index (CPI, base year)",
    "V_26": "CPI of housing, water, fuel, power (base year)",
    "V_27": "Stock market index",
    "V_28": "Population of the city",
    "V_29": "Gold price per ounce (IRR)"
}

ECON_VARS = [
    "Number of building permits issued",
    "Building services index (BSI, base year)",
    "Wholesale price index (WPI, base year)",
    "Total floor area of building permits (m2)",
    "Cumulative liquidity (10^7 IRR)",
    "Private sector investment in new buildings (10^7 IRR)",
    "Land price index (base year, 10^7 IRR)",
    "Number of loans extended by banks",
    "Amount of loans extended (10^7 IRR)",
    "Loan interest rate (%)",
    "Avg construction cost (completion, 10^4 IRR/m2)",
    "Avg construction cost (beginning, 10^4 IRR/m2)",
    "Official exchange rate (IRR/USD)",
    "Unofficial exchange rate (IRR/USD)",
    "Consumer price index (CPI, base year)",
    "CPI of housing, water, fuel, power (base year)",
    "Stock market index",
    "Population of the city",
    "Gold price per ounce (IRR)"
]

df.rename(columns=var_mapping, inplace=True)

currency            = ['Million COPm', 'DOLLARm']
dollar2cop          = 4333.11


def remove_outliers(df, df_encoded_ff):
    # OPTION 3: iqr filter: within 2.22 IQR (equiv. to z-score < 3)
    df_filtered          = df.copy()
    df_orig              = df_encoded_ff.copy()
    df_new               = df.drop('Loan interest rate (%)',axis= 1).copy()

    iqr                  = df_new.quantile(0.75, numeric_only=False) - df_new.quantile(0.25, numeric_only=False)
    lim                  = np.abs((df_new- df_new.median()) / iqr) < 2.22
    cols                 = df_new.select_dtypes('number').columns  # limits to a (float), b (int) and e (timedelta)
    df_orig.loc[:, cols] = df_new.where(lim, np.nan)
    df_orig.dropna(subset=cols, inplace=True) # drop rows with NaN in numerical columns
    df_filtered.loc[:, cols] = df_new.where(lim, np.nan)
    df_filtered.dropna(subset=cols, inplace=True)
    return df_orig, df_filtered

def normalize_by_year(
    df: pd.DataFrame,
    year_col: str,
    cols: List[str],
    agg: Literal["mean", "median"] = "mean",
    suffix: str = "_norm_year"
) -> pd.DataFrame:
    """
    Normaliza columnas numéricas por estadístico anual (promedio/mediana):
        valor_normalizado = valor / estadístico_anual(col)
    - year_col: nombre de la columna de año (p.ej. start_year)
    - cols: lista de columnas a normalizar
    - agg: 'mean' o 'median'
    - suffix: sufijo para las nuevas columnas
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        grp = out.groupby(year_col)[c]
        if agg == "mean":
            stat = grp.transform("mean")
        elif agg == "median":
            stat = grp.transform("median")
        else:
            raise ValueError("agg must be 'mean' or 'median'")

        # Evitar divisiones por 0 o NaN
        stat_safe = stat.replace(0, np.nan)
        out[c + suffix] = out[c] / stat_safe
    return out

def data_preparation_db2(df, currency, dollar2cop):
    """
    Prepara los datos del dataset de Irán:
    - Escala valores según definición del paper (10^4, 10^7, etc.)
    - Convierte a dólares o COP (millones) según el parámetro 'currency'
    """
    df_encoded = pd.get_dummies(df, columns=[
                                            'Loan interest rate (%)'
                                            ])
    
    # Variables que requieren escalamiento inicial (paper IRR units → IRR)
    scale_map = {
        "Preliminary construction cost (beginning, 10^7 IRR)": 1e7,
        "Preliminary construction cost (beginning, 10^4 IRR)": 1e4,
        "Equivalent preliminary cost (base year, 10^4 IRR)": 1e4,
        "Price/unit at beginning (10^4 IRR)": 1e4,
        "Actual sales prices (10^4 IRR) [output]": 1e4,
        "Actual construction costs (10^4 IRR) [output]": 1e4,
        "Cumulative liquidity (10^7 IRR)": 1e7,
        "Private sector investment in new buildings (10^7 IRR)": 1e7,
        "Land price index (base year, 10^7 IRR)": 1e7,
        "Amount of loans extended (10^7 IRR)": 1e7,
        "Avg construction cost (completion, 10^4 IRR/m2)": 1e4,
        "Avg construction cost (beginning, 10^4 IRR/m2)": 1e4,
        "Gold price per ounce (IRR)": 1
    }

    # Aplicar escalamiento inicial
    for col, factor in scale_map.items():
        if col in df.columns:
            df[col] = df[col] * factor

    # Selección de columnas a convertir de IRR → USD o COP
    convert_cols = list(scale_map.keys())

    # Conversión según moneda
    if currency == "DOLLARm":  # Millones de USD
        for col in convert_cols:
            if col in df.columns:
                df[col] = df[col] / df["Official exchange rate (IRR/USD)"]
    elif currency == "Million COPm":  # Millones de COP
        for col in convert_cols:
            if col in df.columns:
                df[col] = ((df[col] / df["Official exchange rate (IRR/USD)"]) * dollar2cop) / 1e6
    
        
    YEAR_COL = "start_year"   # puedes cambiar a "completion_year" si prefieres

    if YEAR_COL not in df.columns:
        raise RuntimeError(f"No se encontró la columna de año '{YEAR_COL}' en el archivo.")
        
    # Normalización por promedio anual
    df_norm_mean = normalize_by_year(df, year_col=YEAR_COL, cols=ECON_VARS, agg="mean", suffix="_norm_year")    
    
    # Seleccionar todas las columnas normalizadas por año
    norm_cols = [c for c in df_norm_mean.columns if c.endswith("_norm_year")]
    
    X = df_norm_mean[norm_cols].fillna(0)
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Probar distintos K
    inertias, silhouettes = [], []
    K_range = range(2, 50)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    
    # Graficar resultados
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Elbow method
    ax[0].plot(K_range, inertias, marker='o')
    ax[0].set_title("Elbow Method (Subset variables)")
    ax[0].set_xlabel("Número de clusters (k)")
    ax[0].set_ylabel("Inercia")
    
    # Silhouette score
    ax[1].plot(K_range, silhouettes, marker='o', color='orange')
    ax[1].set_title("Silhouette Score (Subset variables)")
    ax[1].set_xlabel("Número de clusters (k)")
    ax[1].set_ylabel("Coeficiente de silueta")
    
    plt.tight_layout()
    plt.show()

    # Entrenar modelo final con best_k
    kmeans_final = KMeans(n_clusters=6, random_state=42, n_init=10)
    df_norm_mean["Cluster_ID"] = kmeans_final.fit_predict(X_scaled)
    
    # Centroides en espacio original (valores relativos promedio)
    centroids_scaled = kmeans_final.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    centroids_df = pd.DataFrame(centroids_original, columns=norm_cols)
    centroids_df.index = [f"Cluster_{i+1}" for i in range(6)]
    
    centroids_df.round(2).T
    
    plt.figure(figsize=(10,6))
    sns.heatmap(centroids_df.round(2), annot=True, cmap="coolwarm")
    plt.title("Centroides de clusters (variables normalizadas por año)")
    plt.show()
    
    breakpoint()
        
    breakpoint()
    df_wo_out_encoded, df_wo_out  = remove_outliers(df, df_encoded)    
    
    breakpoint()
    
    return df


gato = data_preparation_db2(df, currency[1], dollar2cop)
