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



# Porcentaje de población urbana
df["porc_urbana"] = df["pob_urbana"] / df["pob_total"]

# Porcentaje de población rural
df["porc_rural"] = df["pob_rural"] / df["pob_total"]

# Residuos por habitante urbano al año en kg
df["residuos_kg_hab_anual"] = (df["qresiduos_dom"] * 1000) / df["pob_urbana"]



# Variación anual de generación de residuos por distrito
df["variacion_anual_residuos"] = df.groupby("ubigeo")["qresiduos_dom"].pct_change()

# Promedio histórico de generación por distrito
df["promedio_residuos_distrito"] = df.groupby("ubigeo")["qresiduos_dom"].transform("mean")

# Categoría de generación
df["categoria_generacion"] = pd.qcut(
    df["qresiduos_dom"],
    q=3,
    labels=["Baja", "Media", "Alta"]
)

print(df.head())