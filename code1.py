import pandas as pd
import numpy as np

# LIMPIEZA

# Cargar dataset principal
df = pd.read_csv("1. DataSet Generación Anual de residuos sólidos domiciliario_Distrital_2014_2024.csv", encoding='latin-1', sep=';')

# Visualizar primeras filas
print(df.head())

# Revisar información general
print(df.info())

# Revisar valores nulos
print(df.isnull().sum())

# Estandarizar nombres de columnas
df.columns = (
    df.columns
    .str.strip() #borra espacios al inicio y al final
    .str.lower() #convierte a minuscula
    .str.replace(" ", "_") #reemplaza espacios por guiones bajos
    .str.replace(".", "", regex=False) #borra puntos
)

print(df.columns)



#------------------NUEVO-------------------#

df["distrito"] = df["distrito"].astype(str).str.strip()
df["distrito"] = df["distrito"].str.replace(r"\s+", " ", regex=True)
df["distrito"] = df["distrito"].str.replace(r"\s*\d+/\s*(?:\d+/\s*)*$", "", regex=True)
df["distrito"] = df["distrito"].str.strip()
#---------------------------------------------

# ANALISIS EXPLORATORIO


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


# VARIABLES DERIVADAS


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

# EXPORTAR A EXCEL
df_ordenado.to_excel(
    "dataset_residuos_limpio_ordenado.xlsx",
    index=False
)

print("Archivo Excel generado correctamente.")

