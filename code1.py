import pandas as pd
import numpy as np
#-----------------------------------------------------------------------------------------------------
# lib para visualización-----------------------------------------------------------------------
import matplotlib.pyplot as plt
import os
#-----------------------------------------------------------------------------------------------------
# libreiajs -------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar dataset limpio 
df = pd.read_csv("DataSet_LIMPIO.csv", encoding='utf-8', sep=';')

# Visualizar primeras filas
print(df.head())

# Revisar información general
print(df.info())

# Revisar valores nulos
print(df.isnull().sum())
print(df.columns)

#-----------------------------------------------------------------------------------------------------------

# ANALISIS EXPLORATORIO-------------------------------------------------------------------------------------



# Estadísticas descriptivas
print(df.describe())

# Revisar años disponibles
print(df["periodo"].unique())

# Generación total por año
generacion_anual = df.groupby("periodo")["qresiduos_dom"].sum()
print(generacion_anual)

# Generación total por departamento
generacion_departamento = df.groupby("departamento")["qresiduos_dom"].sum().sort_values(ascending=False)
print(generacion_departamento)

# Generación promedio por región natural
residuos_por_region = (
    df.groupby("reg_nat")["qresiduos_dom"]
    .mean()
    .sort_values(ascending=False)
)

print(residuos_por_region)

#-----------------------------------------------------------------------------------------------------------------


# VARIABLES DERIVADAS--------------------------------------------------------------------------------------

# Ordenar por distrito y año
df = df.sort_values(["ubigeo", "periodo"])


#cambio para evitar divisiones entre cero
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


# Variación anual de generación de residuos por distrito
df["variacion_anual_residuos"] = df.groupby("ubigeo")["qresiduos_dom"].pct_change()

df["variacion_anual_residuos"] = df["variacion_anual_residuos"].replace(
    [np.inf, -np.inf],
    np.nan
)

# Promedio histórico de generación por distrito
df["promedio_residuos_distrito"] = df.groupby("ubigeo")["qresiduos_dom"].transform("mean")

#--------------cambio-------------#

# Generación acumulada de residuos por distrito
df["generacion_acumulada"] = df.groupby("ubigeo")["qresiduos_dom"].cumsum()

# Generación acumulada previa
df["generacion_acumulada_previa"] = (df["generacion_acumulada"] - df["qresiduos_dom"])

# Generación total histórica por distrito
df["generacion_total_historica"] = (
    df.groupby("ubigeo")["qresiduos_dom"]
    .transform("sum")
)

# Cambio total de residuos entre el primer y último año disponible por distrito
def calcular_cambio_total(serie):
    serie = serie.dropna()
    serie = serie[serie > 0]

    if len(serie) < 2:
        return np.nan

    return (serie.iloc[-1] - serie.iloc[0]) / serie.iloc[0]

df["cambio_total_residuos"] = (
    df.groupby("ubigeo")["qresiduos_dom"]
    .transform(calcular_cambio_total)
)

# Tendencia de crecimiento según el cambio total
df["tendencia_crecimiento"] = np.select(
    [
        df["cambio_total_residuos"] > 0.05,
        df["cambio_total_residuos"] < -0.05
    ],
    [
        "Creciente",
        "Decreciente"
    ],
    default="Estable"
)

# Cuando no hay datos suficientes
df.loc[
    df["cambio_total_residuos"].isna(),
    "tendencia_crecimiento"
] = "Sin datos suficientes"



# Categoría de generación
df["categoria_generacion"] = pd.qcut(
    df["qresiduos_dom"],
    q=3,
    labels=["Baja", "Media", "Alta"]
)

print(df.head())

print(df["categoria_generacion"].value_counts(dropna=False))

# Verificar las nuevas variables
df[
    [
        "ubigeo",
        "distrito",
        "periodo",
        "qresiduos_dom",
        "variacion_anual_residuos",
        "promedio_residuos_distrito",
        "generacion_acumulada",
        "generacion_acumulada_previa",
        "cambio_total_residuos",
        "tendencia_crecimiento",
        "categoria_generacion"
    ]
].head(20)

# ORDENAR FILAS
df = df.sort_values(["departamento", "provincia", "distrito", "periodo"])

# ORDENAR COLUMNAS
columnas_ordenadas = [
    "fecha_corte",
    "periodo",
    "n_sec",
    "ubigeo",
    "reg_nat",
    "departamento",
    "provincia",
    "distrito",
    "pob_total",
    "pob_urbana",
    "pob_rural",
    "gpc_dom",
    "qresiduos_dom",
    "porc_urbana",
    "porc_rural",
    "residuos_kg_hab_anual",
    "variacion_anual_residuos",
    "promedio_residuos_distrito",
    "generacion_acumulada",
    "generacion_acumulada_previa",
    "generacion_total_historica",
    "cambio_total_residuos",
    "tendencia_crecimiento",
    "categoria_generacion"
]

# Mantener solo columnas existentes
columnas_ordenadas = [col for col in columnas_ordenadas if col in df.columns]

df_ordenado = df[columnas_ordenadas]

df_ordenado = df_ordenado.replace([np.inf, -np.inf], np.nan)

# EXPORTAR A EXCEL
df_ordenado.to_excel(
    "dataset_residuos_limpio_ordenado.xlsx",
    index=False
)

print("Archivo Excel generado correctamente.")

#-------------------------------------------------------------------------------------------------------------

# VISUALIZACIÓN DE DATOS--------------------------------------------------------------------------------------

# Crear carpeta para guardar gráficos
os.makedirs("graficos", exist_ok=True)

# 1. Generación total de residuos por año en millones
generacion_anual = df.groupby("periodo")["qresiduos_dom"].sum()

plt.figure(figsize=(8, 5))
plt.plot(generacion_anual.index, generacion_anual.values, marker="o")
plt.title("Generación total de residuos domiciliarios por año")
plt.xlabel("Año")
plt.ylabel("Residuos domiciliarios generados (millones de toneladas)")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficos/generacion_total_por_anio.png")
plt.close()


# 2. Top 10 departamentos con mayor generación acumulada
top_departamentos = (
    df.groupby("departamento")["qresiduos_dom"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(9, 5))
top_departamentos.sort_values().plot(kind="barh")
plt.title("Top 10 departamentos con mayor generación de residuos")
plt.xlabel("Residuos domiciliarios generados")
plt.ylabel("Departamento")
plt.tight_layout()
plt.savefig("graficos/top_10_departamentos.png")
plt.close()


# 3. Generación promedio por región natural
region_promedio = (
    df.groupby("reg_nat")["qresiduos_dom"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(7, 5))
region_promedio.plot(kind="bar")
plt.title("Generación promedio de residuos por región natural")
plt.xlabel("Región natural")
plt.ylabel("Promedio de residuos domiciliarios")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graficos/promedio_region_natural.png")
plt.close()


# 4. Distribución de categorías de generación
conteo_categorias = df["categoria_generacion"].value_counts()

plt.figure(figsize=(7, 5))
conteo_categorias.plot(kind="bar")
plt.title("Cantidad de registros por categoría de generación")
plt.xlabel("Categoría")
plt.ylabel("Cantidad de registros distritales-anuales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graficos/categorias_generacion.png")
plt.close()


# 5. Relación entre población urbana y residuos generados
plt.figure(figsize=(8, 5))
plt.scatter(df["pob_urbana"], df["qresiduos_dom"], alpha=0.5)
plt.title("Relación entre población urbana y residuos domiciliarios")
plt.xlabel("Población urbana")
plt.ylabel("Residuos domiciliarios generados")
plt.tight_layout()
plt.savefig("graficos/poblacion_urbana_vs_residuos.png")
plt.close()


# 6. Relación entre GPC domiciliaria y residuos generados
df_gpc = df[
    (df["gpc_dom"] > 0) &
    (df["qresiduos_dom"] > 0)
]
plt.figure(figsize=(8, 5))
plt.scatter(df_gpc["gpc_dom"], df_gpc["qresiduos_dom"], alpha=0.4, s=15)
plt.yscale("log") # Escala logarítmica para mejor visualización
plt.title("Relación entre GPC domiciliaria y residuos generados")
plt.xlabel("GPC domiciliaria")
plt.ylabel("Residuos domiciliarios generados (escala log)")
plt.tight_layout()
plt.savefig("graficos/gpc_vs_residuos.png")
plt.close()


# 7. Top 15 distritos con mayor generación histórica
top_distritos = (
    df.groupby(["departamento", "provincia", "distrito"])["qresiduos_dom"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 6))
top_distritos.sort_values().plot(kind="barh")
plt.title("Top 15 distritos con mayor generación histórica de residuos")
plt.xlabel("Residuos domiciliarios generados")
plt.ylabel("Distrito")
plt.tight_layout()
plt.savefig("graficos/top_15_distritos.png")
plt.close()


# 8. Distribución del cambio total de residuos
# Dataset único por distrito para analizar cambio total
df_cambio = df[
    [
        "ubigeo",
        "departamento",
        "provincia",
        "distrito",
        "cambio_total_residuos",
        "tendencia_crecimiento"
    ]
].drop_duplicates(subset=["ubigeo"])

df_cambio = df_cambio.dropna(subset=["cambio_total_residuos"])

# Revisar resumen estadístico
print(df_cambio["cambio_total_residuos"].describe())

# Filtrar valores extremos usando percentiles
limite_inferior = df_cambio["cambio_total_residuos"].quantile(0.01)
limite_superior = df_cambio["cambio_total_residuos"].quantile(0.99)

df_cambio_filtrado = df_cambio[
    (df_cambio["cambio_total_residuos"] >= limite_inferior) &
    (df_cambio["cambio_total_residuos"] <= limite_superior)
]

# Graficar en porcentaje
plt.figure(figsize=(8, 5))
plt.hist(df_cambio_filtrado["cambio_total_residuos"] * 100, bins=30)
plt.title("Distribución del cambio total de residuos por distrito")
plt.xlabel("Cambio total de residuos (%)")
plt.ylabel("Cantidad de distritos")
plt.tight_layout()
plt.savefig("graficos/distribucion_cambio_total_corregido.png")
plt.close()


#9. Cantidad de distritos según tendencia de crecimiento
conteo_tendencia = df_cambio["tendencia_crecimiento"].value_counts()

plt.figure(figsize=(7, 5))
conteo_tendencia.plot(kind="bar")
plt.title("Cantidad de distritos según tendencia de crecimiento")
plt.xlabel("Tendencia")
plt.ylabel("Cantidad de distritos")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graficos/tendencia_crecimiento_distritos.png")
plt.close()

# MODELO DE REGRESIÓN --------------------------------------------------------------------------------------




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

os.makedirs("resultados_regresion", exist_ok=True)

# Copia del dataset
df_ml = df.copy()

# Asegurar orden temporal
df_ml = df_ml.sort_values(["ubigeo", "periodo"])

# Asegurar que las variables numéricas estén en formato numérico
columnas_numericas_base = [
    "periodo",
    "pob_total",
    "pob_urbana",
    "pob_rural",
    "gpc_dom",
    "qresiduos_dom",
    "porc_urbana",
    "porc_rural",
    "generacion_acumulada_previa"
]

for col in columnas_numericas_base:
    if col in df_ml.columns:
        df_ml[col] = pd.to_numeric(df_ml[col], errors="coerce")


# VARIABLES HISTÓRICAS SEGURAS ---------------------------------------------------------------------------

# Residuos generados por el distrito en el año anterior
df_ml["residuos_anio_anterior"] = (
    df_ml.groupby("ubigeo")["qresiduos_dom"]
    .shift(1)
)

# GPC del año anterior
df_ml["gpc_anio_anterior"] = (
    df_ml.groupby("ubigeo")["gpc_dom"]
    .shift(1)
)

# Población urbana del año anterior
df_ml["pob_urbana_anio_anterior"] = (
    df_ml.groupby("ubigeo")["pob_urbana"]
    .shift(1)
)

# Variación previa de residuos
df_ml["variacion_residuos_previa"] = (
    df_ml.groupby("ubigeo")["qresiduos_dom"]
    .pct_change()
)

df_ml["variacion_residuos_previa"] = (
    df_ml.groupby("ubigeo")["variacion_residuos_previa"]
    .shift(1)
)

df_ml["variacion_residuos_previa"] = df_ml["variacion_residuos_previa"].replace(
    [np.inf, -np.inf],
    np.nan
)

# Promedio histórico previo
df_ml["conteo_previo"] = df_ml.groupby("ubigeo").cumcount()

df_ml["promedio_residuos_previo"] = np.where(
    df_ml["conteo_previo"] > 0,
    df_ml["generacion_acumulada_previa"] / df_ml["conteo_previo"],
    np.nan
)


# DEFINICIÓN DEL MODELO -----------------------------------------------------------------------------------

objetivo = "qresiduos_dom"

variables_numericas = [
    "periodo",
    "pob_total",
    "pob_urbana",
    "pob_rural",
    "gpc_dom",
    "porc_urbana",
    "porc_rural",
    "residuos_anio_anterior",
    "gpc_anio_anterior",
    "pob_urbana_anio_anterior",
    "variacion_residuos_previa",
    "generacion_acumulada_previa",
    "promedio_residuos_previo"
]

variables_categoricas = [
    "reg_nat",
    "departamento",
    "provincia"
]

columnas_id = [
    "ubigeo",
    "departamento",
    "provincia",
    "distrito",
    "periodo"
]

# Crear lista de columnas sin duplicados
columnas_modelo = columnas_id + variables_numericas + variables_categoricas + [objetivo]
columnas_modelo = list(dict.fromkeys(columnas_modelo))

# Crear dataset del modelo sin columnas duplicadas
df_modelo = df_ml[columnas_modelo].copy()

# Seguridad extra: eliminar columnas duplicadas si existieran
df_modelo = df_modelo.loc[:, ~df_modelo.columns.duplicated()]

# Eliminar filas sin variable objetivo
df_modelo = df_modelo.dropna(subset=[objetivo])



# Eliminar filas sin variable objetivo
df_modelo = df_modelo.dropna(subset=[objetivo])


# DIVISIÓN TEMPORAL ---------------------------------------------------------------------------------------

# Para el trabajo final es mejor probar con el último año disponible
anio_test = df_modelo["periodo"].max()

train_df = df_modelo[df_modelo["periodo"] < anio_test]
test_df = df_modelo[df_modelo["periodo"] == anio_test]

X_train = train_df[variables_numericas + variables_categoricas]
y_train = train_df[objetivo]

X_test = test_df[variables_numericas + variables_categoricas]
y_test = test_df[objetivo]

print("Año usado para prueba:", anio_test)
print("Registros de entrenamiento:", X_train.shape[0])
print("Registros de prueba:", X_test.shape[0])


# PREPROCESAMIENTO ----------------------------------------------------------------------------------------

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocesador = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputador", SimpleImputer(strategy="median")),
                    ("escalador", StandardScaler())
                ]
            ),
            variables_numericas
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputador", SimpleImputer(strategy="most_frequent")),
                    ("onehot", onehot)
                ]
            ),
            variables_categoricas
        )
    ]
)


# MODELOS A COMPARAR --------------------------------------------------------------------------------------

modelos = {
    "Ridge Regression": Ridge(alpha=1.0),

    "Árbol de Decisión": DecisionTreeRegressor(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ENTRENAMIENTO Y EVALUACIÓN ------------------------------------------------------------------------------

resultados = []
modelos_entrenados = {}
predicciones_guardadas = {}

for nombre, algoritmo in modelos.items():

    modelo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", algoritmo)
        ]
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    y_test_np = y_test.to_numpy()
    mascara = y_test_np != 0

    mape = np.mean(
        np.abs((y_test_np[mascara] - y_pred[mascara]) / y_test_np[mascara])
    ) * 100

    resultados.append({
        "modelo": nombre,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE (%)": mape
    })

    modelos_entrenados[nombre] = modelo
    predicciones_guardadas[nombre] = y_pred


df_resultados = pd.DataFrame(resultados).sort_values("RMSE")

print("\nComparación de modelos de regresión:")
print(df_resultados)

df_resultados.to_excel(
    "resultados_regresion/comparacion_modelos_regresion.xlsx",
    index=False
)


# SELECCIÓN DEL MEJOR MODELO ------------------------------------------------------------------------------

mejor_modelo_nombre = df_resultados.iloc[0]["modelo"]
mejor_modelo = modelos_entrenados[mejor_modelo_nombre]
mejor_prediccion = predicciones_guardadas[mejor_modelo_nombre]

print("\nMejor modelo seleccionado:", mejor_modelo_nombre)

# Guardar modelo entrenado
joblib.dump(
    mejor_modelo,
    "resultados_regresion/mejor_modelo_regresion.pkl"
)


# GUARDAR PREDICCIONES ------------------------------------------------------------------------------------

df_predicciones = test_df[columnas_id + [objetivo]].copy()

df_predicciones["prediccion_qresiduos_dom"] = mejor_prediccion

df_predicciones["error_absoluto"] = np.abs(
    df_predicciones[objetivo] - df_predicciones["prediccion_qresiduos_dom"]
)

df_predicciones["error_porcentual"] = np.where(
    df_predicciones[objetivo] > 0,
    df_predicciones["error_absoluto"] / df_predicciones[objetivo] * 100,
    np.nan
)

df_predicciones.to_excel(
    "resultados_regresion/predicciones_mejor_modelo.xlsx",
    index=False
)

print("\nPrimeras predicciones:")
print(df_predicciones.head())


# GRÁFICO: VALORES REALES VS PREDICHOS --------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(y_test, mejor_prediccion, alpha=0.5)

limite_min = min(y_test.min(), mejor_prediccion.min())
limite_max = max(y_test.max(), mejor_prediccion.max())

plt.plot([limite_min, limite_max], [limite_min, limite_max])

plt.title(f"Valores reales vs predichos - {mejor_modelo_nombre}")
plt.xlabel("Valores reales de residuos")
plt.ylabel("Valores predichos de residuos")
plt.tight_layout()
plt.savefig("resultados_regresion/reales_vs_predichos.png")
plt.close()


# GRÁFICO EN ESCALA LOGARÍTMICA ---------------------------------------------------------------------------

df_grafico = df_predicciones[
    (df_predicciones[objetivo] > 0) &
    (df_predicciones["prediccion_qresiduos_dom"] > 0)
]

plt.figure(figsize=(8, 5))
plt.scatter(
    df_grafico[objetivo],
    df_grafico["prediccion_qresiduos_dom"],
    alpha=0.5
)

limite_min = min(
    df_grafico[objetivo].min(),
    df_grafico["prediccion_qresiduos_dom"].min()
)

limite_max = max(
    df_grafico[objetivo].max(),
    df_grafico["prediccion_qresiduos_dom"].max()
)

plt.plot([limite_min, limite_max], [limite_min, limite_max])

plt.xscale("log")
plt.yscale("log")

plt.title(f"Valores reales vs predichos en escala log - {mejor_modelo_nombre}")
plt.xlabel("Valores reales de residuos")
plt.ylabel("Valores predichos de residuos")
plt.tight_layout()
plt.savefig("resultados_regresion/reales_vs_predichos_log.png")
plt.close()


# IMPORTANCIA DE VARIABLES --------------------------------------------------------------------------------

try:
    nombres_variables = mejor_modelo.named_steps["preprocesador"].get_feature_names_out()
    modelo_interno = mejor_modelo.named_steps["modelo"]

    if hasattr(modelo_interno, "feature_importances_"):
        importancias = modelo_interno.feature_importances_

    elif hasattr(modelo_interno, "coef_"):
        importancias = np.abs(modelo_interno.coef_)

    else:
        importancias = None

    if importancias is not None:

        df_importancias = pd.DataFrame({
            "variable": nombres_variables,
            "importancia": importancias
        }).sort_values("importancia", ascending=False)

        print("\nTop 15 variables más importantes:")
        print(df_importancias.head(15))

        df_importancias.to_excel(
            "resultados_regresion/importancia_variables.xlsx",
            index=False
        )

        top_importancias = df_importancias.head(15).sort_values("importancia")

        plt.figure(figsize=(9, 6))
        plt.barh(
            top_importancias["variable"],
            top_importancias["importancia"]
        )
        plt.title("Top 15 variables más importantes del modelo")
        plt.xlabel("Importancia")
        plt.ylabel("Variable")
        plt.tight_layout()
        plt.savefig("resultados_regresion/importancia_variables.png")
        plt.close()

except Exception as e:
    print("No se pudo generar la importancia de variables:", e)


print("\nModelo final de regresión ejecutado correctamente.")


# CLUSTERING POR DISTRITOS ---------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("resultados_clustering", exist_ok=True)

# CAMBIO 1:
# Se ordena por ubigeo y periodo para quedarnos con el nombre más reciente del distrito
df_clust_base = df.sort_values(["ubigeo", "periodo"]).copy()

# CAMBIO 2:
# Se agrupa SOLO por ubigeo para asegurar una sola fila por distrito.
# Antes se agrupaba por ubigeo + departamento + provincia + distrito,
# lo cual podía duplicar distritos si el nombre cambió en algún año.
df_cluster = (
    df_clust_base.groupby("ubigeo")
    .agg(
        departamento=("departamento", "last"),
        provincia=("provincia", "last"),
        distrito=("distrito", "last"),

        pob_total_prom=("pob_total", "mean"),
        porc_urbana_prom=("porc_urbana", "mean"),
        gpc_dom_prom=("gpc_dom", "mean"),
        qresiduos_dom_prom=("qresiduos_dom", "mean"),
        residuos_kg_hab_anual_prom=("residuos_kg_hab_anual", "mean"),

        # CAMBIO 3:
        # Se usa cambio_total_residuos en vez de variacion_anual_prom,
        # porque representa mejor el cambio histórico total del distrito.
        cambio_total_residuos=("cambio_total_residuos", "first")
    )
    .reset_index()
)

variables_cluster = [
    "pob_total_prom",
    "porc_urbana_prom",
    "gpc_dom_prom",
    "qresiduos_dom_prom",
    "residuos_kg_hab_anual_prom",
    "cambio_total_residuos"
]

# Quitar infinitos y distritos sin datos suficientes
df_cluster = df_cluster.replace([np.inf, -np.inf], np.nan)
df_cluster = df_cluster.dropna(subset=variables_cluster)

# CAMBIO 4:
# Filtrado de valores extremos usando cambio_total_residuos
limite_inf = df_cluster["cambio_total_residuos"].quantile(0.01)
limite_sup = df_cluster["cambio_total_residuos"].quantile(0.99)

df_cluster = df_cluster[
    (df_cluster["cambio_total_residuos"] >= limite_inf) &
    (df_cluster["cambio_total_residuos"] <= limite_sup)
]

print("\nDistritos disponibles para clustering:", df_cluster.shape[0])

# Transformación logarítmica en variables con mucha dispersión
df_cluster["log_pob_total"] = np.log1p(df_cluster["pob_total_prom"])
df_cluster["log_qresiduos_dom"] = np.log1p(df_cluster["qresiduos_dom_prom"])
df_cluster["log_residuos_kg_hab"] = np.log1p(df_cluster["residuos_kg_hab_anual_prom"])

variables_modelo_cluster = [
    "log_pob_total",
    "porc_urbana_prom",
    "gpc_dom_prom",
    "log_qresiduos_dom",
    "log_residuos_kg_hab",
    "cambio_total_residuos"
]

# Normalización
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df_cluster[variables_modelo_cluster])

# CAMBIO 5:
# Se guarda tabla de evaluación de K, no solo gráficos
inercias = []
siluetas = []
rango_k = range(2, 8)

for k in rango_k:
    km_prueba = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas_prueba = km_prueba.fit_predict(X_cluster)

    inercias.append(km_prueba.inertia_)
    siluetas.append(silhouette_score(X_cluster, etiquetas_prueba))

df_eval_k = pd.DataFrame({
    "k": list(rango_k),
    "inercia": inercias,
    "silueta": siluetas
})

print("\nEvaluación de K:")
print(df_eval_k)

df_eval_k.to_excel(
    "resultados_clustering/evaluacion_k_clustering.xlsx",
    index=False
)

# Método del codo
plt.figure(figsize=(8, 5))
plt.plot(df_eval_k["k"], df_eval_k["inercia"], marker="o")
plt.title("Método del codo - Clustering de distritos")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.grid(True)
plt.tight_layout()
plt.savefig("resultados_clustering/metodo_codo.png")
plt.close()

# Silhouette
plt.figure(figsize=(8, 5))
plt.plot(df_eval_k["k"], df_eval_k["silueta"], marker="o")
plt.title("Coeficiente de silueta por número de clusters")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Silhouette score")
plt.grid(True)
plt.tight_layout()
plt.savefig("resultados_clustering/silhouette_por_k.png")
plt.close()

# CAMBIO 6:
# Se mantiene K=4, pero ahora queda respaldado por la tabla y los gráficos anteriores.
# Si el profesor pregunta, se justifica por interpretabilidad + codo + silhouette.
k_elegido = 4

kmeans_final = KMeans(
    n_clusters=k_elegido,
    random_state=42,
    n_init=10
)

df_cluster["cluster"] = kmeans_final.fit_predict(X_cluster)

print("\nDistritos por cluster:")
print(df_cluster["cluster"].value_counts())

# Resumen de clusters con variables originales
resumen_clusters = (
    df_cluster.groupby("cluster")[variables_cluster]
    .mean()
    .round(2)
)

resumen_clusters["cantidad_distritos"] = df_cluster.groupby("cluster")["ubigeo"].count()

print("\nResumen de cada cluster:")
print(resumen_clusters)

resumen_clusters.to_excel(
    "resultados_clustering/resumen_clusters.xlsx"
)

# CAMBIO 7:
# Nombres de clusters mejorados según población y cambio histórico
cluster_mayor_poblacion = resumen_clusters["pob_total_prom"].idxmax()
cluster_menor_poblacion = resumen_clusters["pob_total_prom"].idxmin()
cluster_mayor_crecimiento = resumen_clusters["cambio_total_residuos"].idxmax()

nombres_cluster = {}

for c in resumen_clusters.index:

    if c == cluster_mayor_poblacion:
        nombres_cluster[c] = "Grandes urbes"

    elif c == cluster_mayor_crecimiento:
        nombres_cluster[c] = "Crecimiento acelerado"

    elif c == cluster_menor_poblacion:
        nombres_cluster[c] = "Distritos pequeños"

    else:
        nombres_cluster[c] = "Distritos típicos"

df_cluster["nombre_cluster"] = df_cluster["cluster"].map(nombres_cluster)

print("\nNombres asignados a cada cluster:")
print(nombres_cluster)

# Guardar resultado final
df_cluster_resultado = df_cluster[
    [
        "ubigeo",
        "departamento",
        "provincia",
        "distrito",
        "pob_total_prom",
        "porc_urbana_prom",
        "gpc_dom_prom",
        "qresiduos_dom_prom",
        "residuos_kg_hab_anual_prom",
        "cambio_total_residuos",
        "cluster",
        "nombre_cluster"
    ]
]

df_cluster_resultado.to_excel(
    "resultados_clustering/distritos_con_cluster.xlsx",
    index=False
)

# Gráfico: cantidad de distritos por cluster
plt.figure(figsize=(7, 5))
df_cluster["nombre_cluster"].value_counts().plot(kind="bar")
plt.title("Cantidad de distritos por grupo")
plt.xlabel("Grupo")
plt.ylabel("Cantidad de distritos")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("resultados_clustering/distritos_por_cluster.png")
plt.close()

# Gráfico: población vs residuos
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
plt.title("Distritos agrupados por población y generación de residuos")
plt.xlabel("Población total promedio")
plt.ylabel("Residuos domiciliarios promedio")
plt.legend()
plt.tight_layout()
plt.savefig("resultados_clustering/clusters_poblacion_vs_residuos.png")
plt.close()

# Gráfico: residuos por habitante urbano según cluster
plt.figure(figsize=(8, 5))

clusters_ordenados = sorted(df_cluster["cluster"].unique())

datos_boxplot = [
    df_cluster[df_cluster["cluster"] == c]["residuos_kg_hab_anual_prom"]
    for c in clusters_ordenados
]

etiquetas_boxplot = [
    nombres_cluster[c]
    for c in clusters_ordenados
]

plt.boxplot(datos_boxplot, labels=etiquetas_boxplot)
plt.title("Residuos por habitante urbano según grupo de distrito")
plt.ylabel("Residuos (kg/habitante/año)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("resultados_clustering/boxplot_residuos_por_cluster.png")
plt.close()

# CAMBIO 8:
# Se agrega PCA para visualizar los clusters usando todas las variables del modelo
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

df_cluster["pca_1"] = X_pca[:, 0]
df_cluster["pca_2"] = X_pca[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(
    df_cluster["pca_1"],
    df_cluster["pca_2"],
    c=df_cluster["cluster"],
    alpha=0.6
)
plt.title("Visualización de clusters mediante PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.tight_layout()
plt.savefig("resultados_clustering/clusters_pca.png")
plt.close()

print("\n" + "=" * 70)
print("CLUSTERING EJECUTADO CORRECTAMENTE")
print("Resultados guardados en la carpeta 'resultados_clustering/'")
print("=" * 70)