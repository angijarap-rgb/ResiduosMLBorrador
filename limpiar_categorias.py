import pandas as pd
import numpy as np
import unicodedata
import re
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# Cargar dataset
df = pd.read_csv("1. DataSet Generación Anual de residuos sólidos domiciliario_Distrital_2014_2024.csv", encoding='latin-1', sep=';')

# Estandarizar nombres de columnas
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(".", "", regex=False)
)

print("\n" + "=" * 100)
print("LIMPIEZA AUTOMATICA DE CATEGORIAS MAL ESCRITAS")
print("=" * 100)


def normalizar_texto(texto):
    """
    Normaliza texto eliminando acentos y caracteres especiales mal codificados
    """
    if pd.isna(texto):
        return texto

    texto = str(texto).strip()

    # Reemplazar caracteres especiales mal codificados
    reemplazos = {
        '┴': 'Á',
        '═': 'Í',
        'ð': 'Ñ',
        'ë': 'É',
        'ü': 'U',
        'ó': 'O',
        '┐': 'N'
    }

    for mal, bien in reemplazos.items():
        texto = texto.replace(mal, bien)

    # Normalizar unicode (NFC) para manejar caracteres compuestos
    texto = unicodedata.normalize('NFC', texto)

    # Eliminar espacios múltiples
    texto = ' '.join(texto.split())

    # Eliminar sufijos numéricos al final (ej: "NOMBRE 4/", "NOMBRE 16/")
    texto = re.sub(r'\s+\d+/$', '', texto)

    return texto.strip()


def sin_acentos(texto):
    """Devuelve el texto en mayúsculas y sin tildes/diacríticos, para comparar de forma robusta."""
    if pd.isna(texto):
        return texto
    texto = unicodedata.normalize('NFKD', str(texto).upper())
    return ''.join(c for c in texto if not unicodedata.combining(c)).strip()


def normalizar_ubigeo(valor):
    """Devuelve el ubigeo como codigo de 6 digitos para identificar distritos oficiales."""
    if pd.isna(valor):
        return valor

    texto = str(valor).strip()
    if texto.endswith('.0'):
        texto = texto[:-2]

    texto = re.sub(r'\D', '', texto)
    if not texto:
        return np.nan

    return texto.zfill(6)


# ------------------------------------------------------------------
# CORRECCIONES MANUALES ROBUSTAS (no dependen de conocer el texto exacto
# con el que viene mal escrito en el dataset, sino de una comparación
# normalizada sin tildes/puntuación)
# ------------------------------------------------------------------
def correccion_manual_provincia(valor):
    if pd.isna(valor):
        return valor

    base = sin_acentos(valor)
    # quitar puntuación tipo "F." para comparar solo letras
    base_sin_puntos = base.replace('.', '').replace(',', '')
    base_sin_puntos = ' '.join(base_sin_puntos.split())

    # Caso 1: HUANUCO / HUÁNUCO (provincia) -> unificar con tilde
    if base_sin_puntos == 'HUANUCO':
        return 'HUÁNUCO'

    # Caso 2: cualquier variante de "Carlos F. Fitzcarrald" / "Carlos Fermin Fitzcarrald"
    # -> nombre oficial INEI: "Carlos Fermín Fitzcarrald"
    if 'FITZCARRALD' in base_sin_puntos and 'CARLOS' in base_sin_puntos:
        return 'CARLOS FERMÍN FITZCARRALD'

    return valor


# Provincias que NUNCA deben ser eliminadas/fusionadas por el fuzzy matching,
# aunque su similitud con otro nombre supere el umbral. Si detectas más casos
# así en el futuro, solo agrégalos a este set.
PROVINCIAS_PROTEGIDAS = {
    'HUACAYBAMBA',
}

VALORES_PROTEGIDOS_POR_COLUMNA = {
    'provincia': PROVINCIAS_PROTEGIDAS,
}


def limpiar_columna(df, columna):

    print("\n" + "-" * 100)
    print("PROCESANDO: " + columna.upper())
    print("-" * 100)

    # Normalizar todos los valores
    df[columna] = df[columna].apply(normalizar_texto)

    # Mapeo de correcciones de acentos específicas por columna
    correcciones_por_columna = {
        'departamento': {
            'APURIMAC': 'APURÍMAC',
            'HUANUCO': 'HUÁNUCO',
            'JUNIN': 'JUNÍN',
            'SAN MARTIN': 'SAN MARTÍN',
            'ANCASH': 'ÁNCASH'
        },
        'provincia': {
            'ASUNCION': 'ASUNCIÓN',
            'CONCEPCION': 'CONCEPCIÓN',
            'DANIEL ALCIDES CARRION': 'DANIEL ALCIDES CARRIÓN',
            'JUNIN': 'JUNÍN',
            'LA CONVENCION': 'LA CONVENCIÓN',
            'LA UNION': 'LA UNIÓN',
            'MARAÑON': 'MARAÑÓN',
            'NASCA': 'NAZCA',
            'PURUS': 'PURÚS'
        }
    }

    # Aplicar correcciones de acentos según la columna
    cambios_acentos = []
    if columna in correcciones_por_columna:
        for sin_acento, con_acento in correcciones_por_columna[columna].items():
            mascara = df[columna] == sin_acento
            if mascara.any():
                cantidad = mascara.sum()
                df.loc[mascara, columna] = con_acento
                cambios_acentos.append((sin_acento, con_acento, cantidad))

    # Correcciones manuales robustas (Huánuco / Carlos Fermín Fitzcarrald)
    # Se aplican ANTES del fuzzy matching para que esos duplicados queden
    # unificados en un solo valor exacto desde el inicio.
    cambios_manuales = []
    if columna == 'provincia':
        valores_antes = df[columna].copy()
        df[columna] = df[columna].apply(correccion_manual_provincia)
        cambiados = valores_antes != df[columna]
        if cambiados.any():
            for valor_viejo in valores_antes[cambiados].unique():
                cantidad = (valores_antes == valor_viejo).sum()
                valor_nuevo = correccion_manual_provincia(valor_viejo)
                cambios_manuales.append((valor_viejo, valor_nuevo, cantidad))

    # Detectar duplicados similares usando fuzzy matching
    valores_unicos = sorted(df[columna].dropna().unique())
    protegidos = VALORES_PROTEGIDOS_POR_COLUMNA.get(columna, set())
    similares = {}

    for i, val1 in enumerate(valores_unicos):
        if val1 in similares:
            continue

        for val2 in valores_unicos[i + 1:]:
            if val2 in similares:
                continue

            similitud = SequenceMatcher(None, val1, val2).ratio()

            umbral = 0.98 if columna == 'provincia' else 0.90

            if similitud > umbral and val1 != val2:
                val1_protegido = val1 in protegidos
                val2_protegido = val2 in protegidos

                # Si ambos son provincias oficiales protegidas (pero distintas),
                # es un falso positivo del fuzzy matching: NO fusionar.
                if val1_protegido and val2_protegido:
                    continue

                if val1_protegido:
                    si_mas_frecuente, si_menos_frecuente = val1, val2
                elif val2_protegido:
                    si_mas_frecuente, si_menos_frecuente = val2, val1
                else:
                    count1 = (df[columna] == val1).sum()
                    count2 = (df[columna] == val2).sum()
                    si_mas_frecuente = val1 if count1 >= count2 else val2
                    si_menos_frecuente = val2 if count1 >= count2 else val1

                similares[si_menos_frecuente] = si_mas_frecuente

    # Aplicar correcciones
    cambios_realizados = {}

    for val_viejo, val_nuevo in similares.items():
        count = (df[columna] == val_viejo).sum()
        if count > 0:
            df.loc[df[columna] == val_viejo, columna] = val_nuevo

            if val_nuevo not in cambios_realizados:
                cambios_realizados[val_nuevo] = []
            cambios_realizados[val_nuevo].append({
                'viejo': val_viejo,
                'cantidad': count
            })

    # Reportar cambios de acentos
    if cambios_acentos:
        print("\nCORRECCIONES DE ACENTOS APLICADAS:")
        for sin_acento, con_acento, cantidad in cambios_acentos:
            print("  {} registros: '{}' -> '{}'".format(cantidad, sin_acento, con_acento))

    # Reportar correcciones manuales robustas
    if cambios_manuales:
        print("\nCORRECCIONES MANUALES APLICADAS (Huánuco / Carlos Fermín Fitzcarrald):")
        for val_viejo, val_nuevo, cantidad in cambios_manuales:
            print("  {} registros: '{}' -> '{}'".format(cantidad, val_viejo, val_nuevo))

    # Reportar duplicados
    if cambios_realizados:
        print("\nDUPLICADOS SIMILARES CORREGIDOS:")
        for val_standard, cambios in sorted(cambios_realizados.items()):
            print("\n  Valor Standard: '{}'".format(val_standard))
            for cambio in cambios:
                print("    {} registros: '{}' -> '{}'".format(
                    cambio['cantidad'],
                    cambio['viejo'],
                    val_standard
                ))
    else:
        print("\nNo se encontraron duplicados similares")

    if protegidos:
        print("\nVALORES PROTEGIDOS (no se fusionan/eliminan): {}".format(sorted(protegidos)))

    # Estadísticas
    print("\nESTADISTICAS:")
    print("  Valores unicos: {}".format(df[columna].nunique()))
    print("  Registros totales: {}".format(len(df)))

    return df, cambios_realizados


columnas_a_limpiar = ['departamento', 'provincia']

# Para distritos: solo normalizar, sin fuzzy matching
df['distrito'] = df['distrito'].apply(normalizar_texto)
df['_ubigeo_distrito'] = df['ubigeo'].apply(normalizar_ubigeo)
todos_cambios = {}

for columna in columnas_a_limpiar:
    df, cambios = limpiar_columna(df, columna)
    todos_cambios[columna] = cambios


# Resumen final
print("\n" + "=" * 100)
print("RESUMEN FINAL")
print("=" * 100)

print("\nESTADISTICAS FINALES:")
print("  Registros procesados: {}".format(len(df)))
print("  Departamentos unicos: {}".format(df['departamento'].nunique()))
print("  Provincias unicas: {}".format(df['provincia'].nunique()))
print("  Nombres de distrito unicos: {}".format(df['distrito'].nunique()))
print("  Distritos oficiales por ubigeo: {}".format(df['_ubigeo_distrito'].nunique()))

# Los distritos no se deben contar solo por nombre, porque existen homonimos
# en distintas provincias/departamentos. El ubigeo es el identificador oficial.
NUM_DISTRITOS_OFICIALES = 1891
n_distritos = df['_ubigeo_distrito'].nunique()
print("\nVERIFICACION DISTRITOS OFICIALES:")
if n_distritos == NUM_DISTRITOS_OFICIALES:
    print("  OK: Se obtuvieron exactamente {} distritos por ubigeo (esperado).".format(n_distritos))
else:
    print("  ATENCION: Se obtuvieron {} distritos por ubigeo, se esperaban {}.".format(
        n_distritos, NUM_DISTRITOS_OFICIALES))
    print("  Nota: el conteo por nombre puede ser menor por distritos homonimos.")

# Verificación específica del número de provincias oficiales
NUM_PROVINCIAS_OFICIALES = 196
n_provincias = df['provincia'].nunique()
print("\nVERIFICACION PROVINCIAS OFICIALES:")
if n_provincias == NUM_PROVINCIAS_OFICIALES:
    print("  OK: Se obtuvieron exactamente {} provincias (esperado).".format(n_provincias))
else:
    print("  ATENCION: Se obtuvieron {} provincias, se esperaban {}.".format(
        n_provincias, NUM_PROVINCIAS_OFICIALES))
total_cambios = sum(
    len(cambios) for col_cambios in todos_cambios.values() for cambios in col_cambios.values()
)
total_registros_afectados = sum(
    sum(c['cantidad'] for c in cambios)
    for col_cambios in todos_cambios.values()
    for cambios in col_cambios.values()
)

print("\nRESUMEN DE CAMBIOS:")
print("  Total de cambios realizados: {}".format(total_cambios))
print("  Total de registros afectados: {}".format(total_registros_afectados))

# Guardar listado auditado de distritos por ubigeo
distritos_reporte = (
    df[['_ubigeo_distrito', 'departamento', 'provincia', 'distrito']]
    .drop_duplicates(subset=['_ubigeo_distrito'])
    .sort_values('_ubigeo_distrito')
)

with open("distritos_unicos.txt", "w", encoding="utf-8") as f:
    print("LISTA DE DISTRITOS OFICIALES POR UBIGEO", file=f)
    print("Total: {}".format(len(distritos_reporte)), file=f)
    print("", file=f)
    for _, fila in distritos_reporte.iterrows():
        print("{};{};{};{}".format(
            fila['_ubigeo_distrito'],
            fila['departamento'],
            fila['provincia'],
            fila['distrito']
        ), file=f)

# Guardar dataset
print("\n" + "-" * 100)
print("GUARDANDO DATASET LIMPIO")
print("-" * 100)

df = df.drop(columns=['_ubigeo_distrito'])
df.to_csv("DataSet_LIMPIO.csv", encoding='utf-8', sep=';', index=False)
print("\nArchivo guardado: DataSet_LIMPIO.csv")
print("Listado guardado: distritos_unicos.txt")

print("\n" + "=" * 100)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 100 + "\n")
