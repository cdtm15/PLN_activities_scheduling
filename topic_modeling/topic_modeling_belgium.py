#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 17:47:59 2025

@author: cristiantobar
"""

import os
import pandas as pd
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import textwrap
import graphviz
import seaborn as sns
import matplotlib.gridspec as gridspec
import re
import unicodedata
import pandas as pd
import numpy as np
from collections import defaultdict
from graphviz import Digraph
import sympy as sp

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Ruta de la carpeta donde están los archivos Excel
carpeta_excel = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/datos_schedules_construccion"

# Ruta del nuevo archivo Excel que quieres agregar
ruta_apus_traduccion = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/apus_traduccion.xlsx"

# Ruta del archivo excel consolidado de los papers
ruta_consolidado_proj = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/DSLIB_Analysis_Scheet.xlsx"

output_folder_2 = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/procesamiento_lenguaje_natural/petri_net_modular"

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
                                                                               'Actual Start', 
                                                                               'PRC', 
                                                                               'Remaining Duration',
                                                                               'PRC Dev',
                                                                               'Actual Cost', 
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
stops = set(stopwords.words('english'))

stops = stops.union({
    'cm', 'psi', 'cm', 'mpa', 'pvc', 'mts', 'km',
    'incluye', 'mm', 'segun', 'rde', 'mr', 'inc', 'tee', 'pn', 'cal', 'hf',
    'h', 'cms', 'radio', 'dnmm', 'knm', 'awg', 'm', 'd', 'l', 'x', 'sancv',
    'ab', 'gewa', 'tss', 'egw', 'aa', 'hb',  
})
stops = list(stops) # required for later version of CountVectorizer


def quitar_acentos(texto):
    texto = unicodedata.normalize('NFD', texto)                # descompone letras acentuadas
    texto = texto.encode('ascii', 'ignore').decode('utf-8')    # elimina los acentos
    return texto

def limpiar_item(texto):
    texto = texto.lower()                          # convertir a minúsculas
    texto = quitar_acentos(texto)                               # ← aquí aplicas la función
    texto = re.sub(r'\b(no|n°)?\s*\d+\b', '', texto)  # eliminar identificadores como "No 12"
    texto = re.sub(r'\s*([+-])\s*', r' \1 ', texto)
    texto = re.sub(r'\d+(\.\d+)?', '', texto)      # eliminar números (enteros y decimales)
    texto = re.sub(r'[^\w\s]', '', texto)          # eliminar puntuación
    texto = re.sub(r'\s+', ' ', texto).strip()     # eliminar espacios múltiples
    return texto

def plot_top_words(model, feature_names, n_top_words=10):
  fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
  axes = axes.flatten()
  for topic_idx, topic in enumerate(model.components_):
    top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
    top_features = [feature_names[i] for i in top_features_ind]
    weights = topic[top_features_ind]

    ax = axes[topic_idx]
    ax.barh(top_features, weights, height=0.7)
    ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=20)
    for i in "top right left".split():
        ax.spines[i].set_visible(False)
    fig.suptitle('LDA', fontsize=40)

  plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
  plt.show()

def calcular_componentes_test(C, transitions, places):
    C_sym = sp.Matrix(C)
    # -------------------------
    # T-INVARIANTES
    # -------------------------
    nullspace_T = C_sym.nullspace()
    t_invariantes_enteros = []
    t_componentes_nombres = []

    for vec in nullspace_T:
        # Convertir a coeficientes enteros
        lcm = sp.lcm([term.q for term in vec])
        vec_int = [int(lcm * term) for term in vec]

        # Incluir solo si todos los valores son ≥ 0 y alguno > 0
        if all(v >= 0 for v in vec_int) and any(v > 0 for v in vec_int):
            t_invariantes_enteros.append(vec_int)

            if transitions:
                indices = [i for i, val in enumerate(vec_int) if val != 0]
                t_componentes_nombres.append([transitions[i] for i in indices])
    
    # Matriz binaria (1 si transición participa, 0 si no)
    t_invariantes_binarios = []
    for vec in t_invariantes_enteros:
        bin_vec = [1 if v != 0 else 0 for v in vec]
        t_invariantes_binarios.append(bin_vec)
    
    # # -------------------------
    # # P-INVARIANTES
    # # -------------------------
    # C_sym_T = C_sym.T
    # nullspace_P = C_sym_T.nullspace()
    p_componentes_nombres = []

    # for vec in nullspace_P:
    #     lcm = sp.lcm([term.q for term in vec])
    #     vec_int = [int(lcm * term) for term in vec]

    #     if all(v >= 0 for v in vec_int) and any(v > 0 for v in vec_int):
    #         if places:
    #             indices = [i for i, val in enumerate(vec_int) if val != 0]
    #             p_componentes_nombres.append([places[i] for i in indices])
    

    
    return {
        "matriz_C": C,
        "t_invariantes_enteros": t_invariantes_enteros,
        "t_invariantes_binarios": t_invariantes_binarios,
        "t_componentes_nombres": t_componentes_nombres,
        "p_componentes_nombres": p_componentes_nombres
    }


# Extraer indicadores
def extraer_indicadores_por_proyecto(matrices_por_proyecto):

    resumen = []
    for nombre, datos in matrices_por_proyecto.items():
        nombre = datos["filename"]
        pre = datos["Pre"]
        post = datos["Post"]
        places = datos["places"]
        transitions = datos["transitions"]
        
        C = post - pre
        componentes = calcular_componentes_test(C, transitions, places)

        entradas = pre.sum(axis=1)
        salidas = post.sum(axis=1)
        resumen.append({
            "proyecto": nombre,
            "n_places": len(places),
            "n_transitions": len(transitions),
            "n_t_componentes": len(componentes["t_invariantes_binarios"]),
            "n_p_componentes": len(componentes["p_componentes_nombres"]),
            #"n_sifones": len(sif_trap["sifones"]),
            #"n_trampas": len(sif_trap["trampas"]),
            "norm_frobenius_C": np.linalg.norm(post - pre, ord='fro'),
            "solo_entrada_places": int(np.sum((entradas > 0) & (salidas == 0))),
            "solo_salida_places": int(np.sum((salidas > 0) & (entradas == 0))),
            "intermedios_places": int(np.sum((entradas > 0) & (salidas > 0)))
        })
    return pd.DataFrame(resumen)

def parse_relation(rel_str):
    match = re.match(r"(\d+)(FS|SS)([+-]\d+)?(d|w)?", rel_str)
    if match:
        pred_id, rel_type, lag, unit = match.groups()
        lag = lag or ''
        unit = unit or ''
        label = f"{rel_type}{lag}{unit}"
        return pred_id, label
    return None, None

def dibujar_red_petri(pre, post, places, transitions, folder_path, place_labels, nombre_red):
    dot = Digraph(format='pdf')
    dot.attr(rankdir='LR')  # Layout horizontal
    
    
    # Añadir lugares
    for i, place in enumerate(places):
        label = place_labels[place] if place_labels and place in place_labels else place
        dot.node(place, label=label, shape='circle', style='filled', fillcolor='lightblue')

    # Añadir transiciones
    for j, trans in enumerate(transitions):
        dot.node(trans, label=trans, shape='box', style='filled', fillcolor='lightgreen')
        
    # Añadir arcos
    for i, place in enumerate(places):
        for j, trans in enumerate(transitions):
            if pre.iloc[i][j] > 0:
                dot.edge(place, trans)
            if post.iloc[i][j] > 0:
                dot.edge(trans, place)
                
    # Guardar o visualizar
    filename = os.path.join(folder_path, f"{nombre_red}")
    dot.render(filename, cleanup=True)
    print(f"✅ Red guardada como {filename}.pdf")


# Aplicar al DataFrame
df['ITEM_LIMPIO'] = df['Name_x'].astype(str).apply(limpiar_item)

# Mostrar algunos ejemplos
ejemplos = df[['Name_x', 'ITEM_LIMPIO']].drop_duplicates().head(10)
ejemplos


#********Aplicación de LDA para selección de temas¨**************
#Vectorizing information thrugh CounVectorizer
vectorizer = CountVectorizer(stop_words=stops)

X = vectorizer.fit_transform(df['ITEM_LIMPIO'])
#
lda = LatentDirichletAllocation(
    n_components=10, # default: 10
    random_state=12345,
)

lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
plot_top_words(lda, feature_names);

# --- Paso 5: Mostrar los temas ---
def mostrar_top_palabras(modelo, feature_names, n_palabras=10):
    temas = []
    for idx, topic in enumerate(modelo.components_):
        palabras = [feature_names[i] for i in topic.argsort()[:-n_palabras - 1:-1]]
        temas.append((f"Tema {idx+1}", palabras))
    return temas

vocabulario = vectorizer.get_feature_names_out()
temas = mostrar_top_palabras(lda, vocabulario, n_palabras=10)

temas_df = pd.DataFrame(temas, columns=['TEMA', 'PALABRAS_CLAVE'])
temas_df

Z = lda.transform(X)

# Convertir a DataFrame con nombres de columnas "TEMA_1", "TEMA_2", ..., "TEMA_10"
num_topics = Z.shape[1]
column_names = [f'TEMA_{i+1}' for i in range(num_topics)]
df_lda = pd.DataFrame(Z, columns=column_names)

#Hacer merge por índice (asumiendo que el orden de documentos se conserva)
df_resultado_maestro = pd.concat([df.reset_index(drop=True), df_lda.reset_index(drop=True)], axis=1)

# Diccionario de etiquetas personalizadas para los temas
tema_labels = {
    "TEMA_1": "MEP Installations and Vertical Circulation",
    "TEMA_2": "Interior Fit-Out and Joinery",
    "TEMA_3": "Wet Areas and Final Touches",
    "TEMA_4": "Interior Enclosures and HVAC",
    "TEMA_5": "General Finishing and Site Configuration",
    "TEMA_6": "Envelope and Structural Roofing",
    "TEMA_7": "Exterior Architectural Finishes",
    "TEMA_8": "Sanitary Installations and Early Works",
    "TEMA_9": "Structural Masonry and Base Finishes",
    "TEMA_10": "Concrete and Substructure Works"
}

column_order = ["Concrete and Substructure Works",
                "Structural Masonry and Base Finishes",
                "General Finishing and Site Configuration",
                "Sanitary Installations and Early Works",
                "MEP Installations and Vertical Circulation",
                "Envelope and Structural Roofing",
                "Interior Enclosures and HVAC",
                "Interior Fit-Out and Joinery",
                "Wet Areas and Final Touches",
                "Exterior Architectural Finishes",
                ]


# # 1. Crear una nueva columna con el tema dominante por documento
# df_resultado_maestro['TEMA_DOMINANTE'] = df_resultado_maestro[[f'TEMA_{i+1}' for i in range(10)]].idxmax(axis=1)

df_resultado_maestro['TEMA_DOMINANTE_COD'] = df_resultado_maestro[[f'TEMA_{i+1}' for i in range(10)]].idxmax(axis=1)

# 2. Mapear al nombre descriptivo
df_resultado_maestro['TEMA_DOMINANTE'] = df_resultado_maestro['TEMA_DOMINANTE_COD'].map(tema_labels)

# 2. Agrupar por Project_ID y tema dominante, y contar
tema_por_proyecto = df_resultado_maestro.groupby(['Filename','Project_ID', 'TEMA_DOMINANTE']).size().unstack(fill_value=0)

# 3. (Opcional) Normalizar para ver proporciones por proyecto
tema_por_proyecto_pct = tema_por_proyecto.div(tema_por_proyecto.sum(axis=1), axis=0)

# 1. Agrupar por Project_ID y sacar promedio de los temas
#tema_avg_por_proyecto = df_resultado.groupby('Project_ID')[[f'TEMA_{i+1}' for i in range(10)]].mean()




# Reordenamos las columnas para que los temas estén en orden del 1 al 10
#column_order = ['TEMA_' + str(i) for i in range(1, 11)]
df_temas_2 = tema_por_proyecto_pct[column_order]
df_temas_2 = df_temas_2.rename(columns=tema_labels)

df_temas = tema_por_proyecto_pct

# Convertimos a formato largo para graficar con Seaborn
df_long = df_temas.melt(var_name='Tema', value_name='Pertenencia')

# Gráfico tipo boxplot
plt.figure(figsize=(12, 10))
sns.boxplot(x='Tema', y='Pertenencia', data=df_long, palette='viridis')
plt.title('Distribución de pertenencia temática a lo largo de todos los proyectos')
plt.ylabel('Pertenencia')
plt.xlabel('Tema')
plt.xticks(rotation=80)
plt.grid(True)
plt.tight_layout()
plt.show()

# Configuración del heatmap
plt.figure(figsize=(14, 16))
sns.heatmap(df_temas, cmap='YlGnBu', annot=False, cbar_kws={'label': 'Pertenencia temática'}, linewidths=0.5)
# Personalización del gráfico
plt.title('Mapa de calor de pertenencia temática por proyecto', fontsize=14)
plt.xlabel('Tema', fontsize=12)
plt.ylabel('Proyecto (Project_ID)', fontsize=12)
# plt.yticks(ticks=range(len(df_temas)), labels=tema_por_proyecto_pct['Filename'], rotation=0, fontsize=8)
plt.xticks(rotation=80)
plt.tight_layout()
plt.show()

# Configuración del heatmap
plt.figure(figsize=(14, 16))
sns.heatmap(df_temas_2, cmap='YlGnBu', annot=False, cbar_kws={'label': 'Pertenencia temática'}, linewidths=0.5)
# Personalización del gráfico
plt.title('Mapa de calor de pertenencia temática por proyecto', fontsize=14)
plt.xlabel('Tema', fontsize=12)
plt.ylabel('Proyecto (Project_ID)', fontsize=12)
# plt.yticks(ticks=range(len(df_temas)), labels=tema_por_proyecto_pct['Filename'], rotation=0, fontsize=8)
plt.xticks(rotation=80)
plt.tight_layout()
plt.show()


petri_data_por_proyecto = {}  # Diccionario principal

# Iterar por cada proyecto
for project_id in df_resultado_maestro['Project_ID'].unique():
    df_project = df_resultado_maestro[df_resultado_maestro['Project_ID'] == project_id].copy()

    filename = df_project['Filename'].unique()[0]
    relaciones_temas = []

    for _, row in df_project.iterrows():
        tema_destino = row['TEMA_DOMINANTE']
        predecesores = str(row['Predecessors']).split(';') if pd.notna(row['Predecessors']) else []

        for rel in predecesores:
            pred_id, tipo_rel = parse_relation(rel)
            if pred_id and tipo_rel:
                fila_pred = df_project[df_project['ID'] == int(pred_id)]
                if not fila_pred.empty:
                    tema_origen = fila_pred['TEMA_DOMINANTE'].values[0]
                    relaciones_temas.append((tema_origen, tema_destino, tipo_rel))

    if not relaciones_temas:
        print(f"⚠️ No se encontraron relaciones válidas para el Project_ID = {project_id}")
        continue

    temas = sorted(df_project['TEMA_DOMINANTE'].unique())
    plazas = {f"P{tema}": tema for tema in temas}
    transiciones = [f"T{i}" for i in range(len(relaciones_temas))]

    pre_matrix = pd.DataFrame(0, index=plazas.keys(), columns=transiciones)
    post_matrix = pd.DataFrame(0, index=plazas.keys(), columns=transiciones)

    for i, (tema_origen, tema_destino, tipo_rel) in enumerate(relaciones_temas):
        pre_matrix.at[f"P{tema_origen}", f"T{i}"] = 1
        post_matrix.at[f"P{tema_destino}", f"T{i}"] = 1

    # Guardar en el diccionario
    petri_data_por_proyecto[project_id] = {
        'filename': filename,
        'Pre': pre_matrix,
        'Post': post_matrix,
        'places': plazas,
        'transitions': transiciones,
        'relaciones_temas': relaciones_temas,
        'temas': temas
    }
    
    # df_indicadores = extraer_indicadores_por_proyecto(petri_data_por_proyecto)
    
    # # Dibujar red de Petri
    # nombre_red = f"{filename}_Red_Project"
    # dibujar_red_petri(pre_matrix, post_matrix, plazas, transiciones, output_folder_2, temas, nombre_red)


# #***************Empleo de Cosine Similarity para clustering***************
# # TF-IDF vectorization
# vectorizer = TfidfVectorizer(stop_words=stops)
# X_tfidf = vectorizer.fit_transform(df['ITEM_LIMPIO'])

# # Calcular matriz de similitud coseno
# similitud = cosine_similarity(X_tfidf)

# # Convertir a DataFrame para visualización (primeros 10 ítems)
# similitud_df = pd.DataFrame(similitud, index=df['ITEM_LIMPIO'].values, columns=df['ITEM_LIMPIO'].values)

# # Mostrar matriz de similitud entre los primeros 10 ítems
# similitud_df.iloc[:10, :10]

# # Usamos la matriz de similitud para generar una matriz de distancias
# # (1 - similitud) convierte similitud coseno en distancia
# distancia = 1 - similitud

# # Aplicamos clustering jerárquico aglomerativo
# n_clusters = 10  # Puedes ajustar este número según el caso
# modelo_clustering = AgglomerativeClustering(
#     n_clusters=n_clusters,
#     linkage='average'
# )

# # Ajustar el modelo a la matriz de distancias
# labels = modelo_clustering.fit_predict(distancia)

# # Agregar la etiqueta de cluster al dataframe
# df['CLUSTER'] = labels

# # Mostrar algunos ejemplos por cluster
# grupos_ejemplo = df.groupby('CLUSTER').head(3).sort_values(by='CLUSTER')

# # Agrupar todos los ítems del cluster en una tabla
# resumen_cluster = df.groupby('CLUSTER')['ITEM_LIMPIO'].apply(lambda x: '\n- '.join(x.head(5))).reset_index()
# resumen_cluster.columns = ['CLUSTER', 'EJEMPLOS_DE_ITEMS']

# # Contar actividades por clúster y graficar
# conteo_clusters_edificaciones = df['CLUSTER'].value_counts().sort_index()

# plt.figure(figsize=(10, 5))
# sns.barplot(x=conteo_clusters_edificaciones.index, y=conteo_clusters_edificaciones.values, palette='viridis')
# plt.title('Número de actividades por clúster')
# plt.xlabel('Clúster')
# plt.ylabel('Cantidad de actividades')
# plt.tight_layout()
# plt.show()

# # Calcular perplejidad para distintos números de temas
# rango_temas = list(range(2, 100))
# perplejidades = []

# for n in rango_temas:
#     lda = LatentDirichletAllocation(n_components=n, random_state=42)
#     lda.fit(X)
#     perplexity = lda.perplexity(X)
#     perplejidades.append(perplexity)

# # Graficar resultados
# plt.figure(figsize=(8, 5))
# plt.plot(rango_temas, perplejidades, marker='o', linestyle='-')
# plt.title('Perplejidad vs Número de temas en LDA')
# plt.xlabel('Número de temas')
# plt.ylabel('Perplejidad')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# dist_array = 1 - similitud
# mejores_k = []
# sil_scores = []

# for k in range(2, 100):
#     modelo = AgglomerativeClustering(n_clusters=k, linkage='average')
#     etiquetas = modelo.fit_predict(dist_array)
#     sil = silhouette_score(1 - dist_array, etiquetas)
#     mejores_k.append(k)
#     sil_scores.append(sil)

# # Graficar resultados
# plt.plot(mejores_k, sil_scores, marker='o')
# plt.title('Silhouette Score vs Número de Clústeres')
# plt.xlabel('Número de Clústeres')
# plt.ylabel('Silhouette Score')
# plt.grid(True)
# plt.show()
