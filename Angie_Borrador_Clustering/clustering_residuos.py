import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# carga del dataset


RUTA_DATA = os.path.join("data", "DataSet_LIMPIO.csv")

df = pd.read_csv(RUTA_DATA, encoding="utf-8", sep=";")

print("Dataset cargado:", df.shape)
print(df.head(3))

# Por si las columnas no vinieran ya estandarizadas
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(".", "", regex=False)
)

df = df.sort_values(["ubigeo", "periodo"])


# variables derivadas

df["porc_urbana"] = np.where(
    df["pob_total"] > 0,
    df["pob_urbana"] / df["pob_total"],
    np.nan
)

df["porc_rural"] = np.where(
    df["pob_total"] > 0,
    df["pob_rural"] / df["pob_total"],
    np.nan
)

df["residuos_kg_hab_anual"] = np.where(
    df["pob_urbana"] > 0,
    (df["qresiduos_dom"] * 1000) / df["pob_urbana"],
    np.nan
)

df["variacion_anual_residuos"] = df.groupby("ubigeo")["qresiduos_dom"].pct_change()
df["variacion_anual_residuos"] = df["variacion_anual_residuos"].replace(
    [np.inf, -np.inf], np.nan
)

df["promedio_residuos_distrito"] = df.groupby("ubigeo")["qresiduos_dom"].transform("mean")

print("\nVariables derivadas calculadas correctamente.")
print(df[["ubigeo", "distrito", "periodo", "qresiduos_dom",
          "porc_urbana", "residuos_kg_hab_anual"]].head())



# clustering por distritos

os.makedirs("resultados_clustering", exist_ok=True)

# Construcción una fila por distrito (promedio historico 2014-2024)
df_cluster = (
    df.groupby(["ubigeo", "departamento", "provincia", "distrito"])
    .agg(
        pob_total_prom=("pob_total", "mean"),
        porc_urbana_prom=("porc_urbana", "mean"),
        qresiduos_dom_prom=("qresiduos_dom", "mean"),
        residuos_kg_hab_anual_prom=("residuos_kg_hab_anual", "mean"),
        variacion_anual_prom=("variacion_anual_residuos", "mean")
    )
    .reset_index()
)

variables_cluster = [
    "pob_total_prom",
    "porc_urbana_prom",
    "qresiduos_dom_prom",
    "residuos_kg_hab_anual_prom",
    "variacion_anual_prom"
]

# Quitar distritos sin datos suficientes
df_cluster = df_cluster.dropna(subset=variables_cluster)

# Quitar valores extremos de variacion anual (distritos con datos atipicos)
limite_inf = df_cluster["variacion_anual_prom"].quantile(0.01)
limite_sup = df_cluster["variacion_anual_prom"].quantile(0.99)
df_cluster = df_cluster[
    (df_cluster["variacion_anual_prom"] >= limite_inf) &
    (df_cluster["variacion_anual_prom"] <= limite_sup)
]

print("\nDistritos disponibles para clustering:", df_cluster.shape[0])

# Transformacion logaritmica en variables con mucha dispersion
df_cluster["log_pob_total"] = np.log1p(df_cluster["pob_total_prom"])
df_cluster["log_qresiduos_dom"] = np.log1p(df_cluster["qresiduos_dom_prom"])
df_cluster["log_residuos_kg_hab"] = np.log1p(df_cluster["residuos_kg_hab_anual_prom"])

variables_modelo_cluster = [
    "log_pob_total",
    "porc_urbana_prom",
    "log_qresiduos_dom",
    "log_residuos_kg_hab",
    "variacion_anual_prom"
]

# Normalizacion
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df_cluster[variables_modelo_cluster])

# Metodo del codo + silhouette para elegir K
inercias = []
siluetas = []
rango_k = range(2, 8)

for k in rango_k:
    km_prueba = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas_prueba = km_prueba.fit_predict(X_cluster)
    inercias.append(km_prueba.inertia_)
    siluetas.append(silhouette_score(X_cluster, etiquetas_prueba))

plt.figure(figsize=(8, 5))
plt.plot(list(rango_k), inercias, marker="o")
plt.title("Metodo del codo - Clustering de distritos")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("Inercia")
plt.grid(True)
plt.tight_layout()
plt.savefig("resultados_clustering/metodo_codo.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(list(rango_k), siluetas, marker="o", color="orange")
plt.title("Coeficiente de silueta por numero de clusters")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("Silhouette score")
plt.grid(True)
plt.tight_layout()
plt.savefig("resultados_clustering/silhouette_por_k.png")
plt.close()

# K=4 ofrece grupos interpretables y balanceados para este dataset
k_elegido = 4

kmeans_final = KMeans(n_clusters=k_elegido, random_state=42, n_init=10)
df_cluster["cluster"] = kmeans_final.fit_predict(X_cluster)

print("\nDistritos por cluster:")
print(df_cluster["cluster"].value_counts())

#  promedio de cada variable por cluster
resumen_clusters = (
    df_cluster.groupby("cluster")[variables_cluster]
    .mean()
    .round(2)
)

print("\nResumen de cada cluster (variables originales):")
print(resumen_clusters)

resumen_clusters.to_excel("resultados_clustering/resumen_clusters.xlsx")

# Asignar nombre descriptivo segun el patron de cada cluster
nombres_cluster = {}
for c in resumen_clusters.index:
    pob = resumen_clusters.loc[c, "pob_total_prom"]
    var = resumen_clusters.loc[c, "variacion_anual_prom"]

    if pob == resumen_clusters["pob_total_prom"].max():
        nombres_cluster[c] = "Grandes urbes"
    elif var == resumen_clusters["variacion_anual_prom"].max():
        nombres_cluster[c] = "Crecimiento acelerado"
    elif pob == resumen_clusters["pob_total_prom"].min():
        nombres_cluster[c] = "Distritos pequenos"
    else:
        nombres_cluster[c] = "Distritos tipicos"

df_cluster["nombre_cluster"] = df_cluster["cluster"].map(nombres_cluster)

print("\nNombres asignados a cada cluster:")
print(nombres_cluster)

# Guardar resultado final con distrito + cluster asignado
df_cluster_resultado = df_cluster[
    [
        "ubigeo", "departamento", "provincia", "distrito",
        "pob_total_prom", "porc_urbana_prom", "qresiduos_dom_prom",
        "residuos_kg_hab_anual_prom", "variacion_anual_prom",
        "cluster", "nombre_cluster"
    ]
]

df_cluster_resultado.to_excel(
    "resultados_clustering/distritos_con_cluster.xlsx",
    index=False
)

# GRAFICO: distribucion de distritos por cluster
plt.figure(figsize=(7, 5))
df_cluster["nombre_cluster"].value_counts().plot(kind="bar")
plt.title("Cantidad de distritos por grupo (cluster)")
plt.xlabel("Grupo")
plt.ylabel("Cantidad de distritos")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("resultados_clustering/distritos_por_cluster.png")
plt.close()

# Grafico - dispersion poblacion vs residuos coloreado por cluster
plt.figure(figsize=(8, 6))
for c in sorted(df_cluster["cluster"].unique()):
    subset = df_cluster[df_cluster["cluster"] == c]
    plt.scatter(
        subset["pob_total_prom"],
        subset["qresiduos_dom_prom"],
        label=nombres_cluster[c],
        alpha=0.6
    )

plt.xscale("log")
plt.yscale("log")
plt.title("Distritos agrupados por poblacion y generacion de residuos")
plt.xlabel("Poblacion total promedio (escala log)")
plt.ylabel("Residuos domiciliarios promedio (escala log)")
plt.legend()
plt.tight_layout()
plt.savefig("resultados_clustering/clusters_poblacion_vs_residuos.png")
plt.close()

# Grafco - residuos per capita por cluster
plt.figure(figsize=(8, 5))
datos_boxplot = [
    df_cluster[df_cluster["cluster"] == c]["residuos_kg_hab_anual_prom"]
    for c in sorted(df_cluster["cluster"].unique())
]
etiquetas_boxplot = [nombres_cluster[c] for c in sorted(df_cluster["cluster"].unique())]

plt.boxplot(datos_boxplot, tick_labels=etiquetas_boxplot)
plt.title("Residuos por habitante urbano segun grupo de distrito")
plt.ylabel("Residuos (kg/habitante/anio)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("resultados_clustering/boxplot_residuos_por_cluster.png")
plt.close()

print("\n" + "=" * 70)
print("CLUSTERING EJECUTADO CORRECTAMENTE")
print("Resultados guardados en la carpeta 'resultados_clustering/'")
print("=" * 70)