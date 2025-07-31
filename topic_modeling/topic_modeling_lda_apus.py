#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:41:53 2025

@author: cristiantobar
"""

import pandas as pd
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


from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


#**************MAIN******************
nltk.download('stopwords')

url_apus_building = '/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/apus_construccion_uni.xlsx'
url_apus = '/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/apus_traduccion.xlsx'
url_apus_esp = '/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/apus_construccion_colombia.xlsx'

language = ['spanish','english']
selec_lang = language[1]

if selec_lang == 'spanish':
    df = pd.read_excel(url_apus_esp)
    stops = set(stopwords.words('spanish'))

if selec_lang == 'english':
    df = pd.read_excel(url_apus)
    stops = set(stopwords.words('english'))

stops = stops.union({
    'cm', 'psi', 'cm', 'mpa', 'pvc', 'mts', 'km',
    'incluye', 'mm', 'segun', 'rde', 'mr', 'inc', 'tee', 'pn', 'cal', 'hf',
    'h', 'cms', 'radio', 'dnmm', 'knm', 'awg', 'm', 'd', 'l', 'x'
})
stops = list(stops) # required for later version of CountVectorizer

# Crear un conjunto de relaciones únicas entre capítulos y subcapítulos
cap_subcap_pairs = df[['CAPITULO', 'SUBCAPITULO']].drop_duplicates()


# Obtener los capítulos únicos
capitulos = cap_subcap_pairs['CAPITULO'].unique()

# Crear el grafo
G = nx.DiGraph()

# Nodo raíz
root = "Capítulos de obra"
G.add_node(root)

# Agregar relaciones desde el nodo raíz a los capítulos
for cap in capitulos:
    G.add_node(cap)
    G.add_edge(root, cap)

# Generar layout
pos = nx.spring_layout(G, seed=42)

# Dibujar el grafo
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray',
        node_size=3000, font_size=10, font_weight='bold', arrows=True)
plt.title("Mapa conceptual de Capítulos de Obra", fontsize=14)
plt.tight_layout()
plt.show()


# Filtrar columnas necesarias y limpiar
df_filtered = df[['CAPITULO', 'SUBCAPITULO', 'TOTAL', 'ITEM']].dropna()
df_filtered['TOTAL'] = pd.to_numeric(df_filtered['TOTAL'], errors='coerce')

# Obtener capítulos únicos
capitulos = df_filtered['CAPITULO'].unique()

# Crear gráficos por capítulo
for cap in capitulos:
    df_cap = df_filtered[df_filtered['CAPITULO'] == cap]

    # Ordenar subcapítulos por mediana del TOTAL
    orden_subcap = df_cap.groupby('SUBCAPITULO')['TOTAL'].median().sort_values().index
    df_cap['SUBCAPITULO'] = pd.Categorical(df_cap['SUBCAPITULO'], categories=orden_subcap, ordered=True)
    
    # Conteo de ítems por subcapítulo
    item_counts = df_cap.groupby('SUBCAPITULO')['ITEM'].count().reindex(orden_subcap)
    
    # Crear figura con dos subplots alineados horizontalmente
    fig = plt.figure(figsize=(14, max(6, len(orden_subcap) * 0.35)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)

    # Subplot 1: Boxplot
    ax0 = plt.subplot(gs[0])
    sns.boxplot(data=df_cap, y='SUBCAPITULO', x='TOTAL', ax=ax0, orient='h', palette='Set2')
    ax0.set_title(f'{cap} - Distribución del valor total')
    ax0.set_ylabel('Subcapítulo')
    ax0.set_xlabel('Valor total')

    # Subplot 2: Conteo de ítems
    ax1 = plt.subplot(gs[1], sharey=ax0)
    sns.barplot(x=item_counts.values, y=item_counts.index, ax=ax1)
    ax1.set_title('Número de Actividades')
    ax1.set_xlabel('Cantidad')
    ax1.set_ylabel('')
    ax1.set_xlim(0, 100)
    ax1.tick_params(left=False, labelleft=False)

    plt.tight_layout()
    plt.show()

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

# Aplicar al DataFrame
df['ITEM_LIMPIO'] = df['ITEM'].astype(str).apply(limpiar_item)

if selec_lang == 'spanish':
    df_edificaciones = df[df['CAPITULO'] == 'EDIFICACIONES']
    
if selec_lang == 'english':
    df_edificaciones = df[df['CAPITULO'] == 'BUILDINGS']

# Mostrar algunos ejemplos
ejemplos = df[['ITEM', 'ITEM_LIMPIO']].drop_duplicates().head(10)
ejemplos

# Agrupar todos los textos de ITEM_LIMPIO por CAPITULO
items_por_capitulo = df.groupby('CAPITULO')['ITEM_LIMPIO'].apply(lambda textos: ' '.join(textos)).reset_index()

# Renombrar la columna resultante
items_por_capitulo.columns = ['CAPITULO', 'TEXTO_CONCATENADO']

# Agrupar todos los textos de ITEM_LIMPIO por CAPITULO
items_por_subcapitulo = df.groupby('SUBCAPITULO')['ITEM_LIMPIO'].apply(lambda textos: ' '.join(textos)).reset_index()

# Renombrar la columna resultante
items_por_subcapitulo.columns = ['SUBCAPITULO', 'TEXTO_CONCATENADO']
    

#********Aplicación de LDA para selección de temas¨**************
#Vectorizing information thrugh CounVectorizer
vectorizer = CountVectorizer(stop_words=stops)

#X = vectorizer.fit_transform(items_por_subcapitulo['TEXTO_CONCATENADO'])

X = vectorizer.fit_transform(df_edificaciones['ITEM_LIMPIO'])
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

#***************Empleo de Cosine Similarity para clustering***************
# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words=stops)
X_tfidf = vectorizer.fit_transform(df_edificaciones['ITEM_LIMPIO'])

# Calcular matriz de similitud coseno
similitud = cosine_similarity(X_tfidf)

# Convertir a DataFrame para visualización (primeros 10 ítems)
similitud_df = pd.DataFrame(similitud, index=df_edificaciones['ITEM_LIMPIO'].values, columns=df_edificaciones['ITEM_LIMPIO'].values)

# Mostrar matriz de similitud entre los primeros 10 ítems
similitud_df.iloc[:10, :10]

# Usamos la matriz de similitud para generar una matriz de distancias
# (1 - similitud) convierte similitud coseno en distancia
distancia = 1 - similitud

# Aplicamos clustering jerárquico aglomerativo
n_clusters = 10  # Puedes ajustar este número según el caso
modelo_clustering = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='average'
)

# Ajustar el modelo a la matriz de distancias
labels = modelo_clustering.fit_predict(distancia)

# Agregar la etiqueta de cluster al dataframe
df_edificaciones['CLUSTER'] = labels

# Mostrar algunos ejemplos por cluster
grupos_ejemplo = df_edificaciones.groupby('CLUSTER').head(3).sort_values(by='CLUSTER')

# Agrupar todos los ítems del cluster en una tabla
resumen_cluster = df_edificaciones.groupby('CLUSTER')['ITEM_LIMPIO'].apply(lambda x: '\n- '.join(x.head(5))).reset_index()
resumen_cluster.columns = ['CLUSTER', 'EJEMPLOS_DE_ITEMS']

# Contar actividades por clúster y graficar
conteo_clusters_edificaciones = df_edificaciones['CLUSTER'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
sns.barplot(x=conteo_clusters_edificaciones.index, y=conteo_clusters_edificaciones.values, palette='viridis')
plt.title('Número de actividades por clúster en capítulo EDIFICACIONES')
plt.xlabel('Clúster')
plt.ylabel('Cantidad de actividades')
plt.tight_layout()
plt.show()

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