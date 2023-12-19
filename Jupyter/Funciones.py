## FUNCIONES DE UTILIDAD PARA EL ETL Y EDA
# Importaciones
import pandas as pd
from textblob import TextBlob
import re
from datetime import datetime


# Funciones
def verificar_tipo_datos(df):
    '''
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna, el porcentaje de valores no nulos y nulos, así como la
    cantidad de valores nulos por columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
        - 'porcentaje_no_nulos': Porcentaje de valores no nulos en cada columna.
        - 'porcentaje_nulos': Porcentaje de valores nulos en cada columna.
        - 'nulos': Cantidad de valores nulos en cada columna.
    '''

    mi_dict = {"nombre": [], "tipo_datos": [], "porcentaje_no_nulos": [], "porcentaje_nulos": [], "nulos": []}

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        mi_dict["nombre"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
        mi_dict["porcentaje_no_nulos"].append(round(porcentaje_no_nulos, 2))
        mi_dict["porcentaje_nulos"].append(round(100-porcentaje_no_nulos, 2))
        mi_dict["nulos"].append(df[columna].isnull().sum())

    df_info = pd.DataFrame(mi_dict)
        
    return df_info



#Se convierten las listas en tuplas para verificar los duplicados

def filas_duplicadas(dataframe):
    """
    Verifica si hay filas duplicadas en un DataFrame de Pandas.

    Parameters:
    - dataframe (pd.DataFrame): El DataFrame que se va a verificar.

    Returns:
    - bool: True si hay al menos una fila duplicada, False si no hay filas duplicadas.
    """
    
    dataframe = dataframe.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
    
   
    duplicados = dataframe.duplicated()
    
    return any(duplicados)



# Se eliminan duplicados
def eliminar_filas_duplicadas(dataframe):
    """
    Elimina las filas duplicadas de un DataFrame de Pandas.

    Parameters:
    - dataframe (pd.DataFrame): El DataFrame del cual eliminar las filas duplicadas.

    Returns:
    - pd.DataFrame: Un nuevo DataFrame sin filas duplicadas.
    """
    dataframe = dataframe.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
    dataframe_sin_duplicados = dataframe.drop_duplicates()
    
    return dataframe_sin_duplicados


def duplicados_por_columna(df, columna):
    
    """
    Encuentra y muestra las filas duplicadas en un DataFrame de Pandas basándose en una columna específica.

    Parameters:
    - df (pd.DataFrame): El DataFrame en el que buscar filas duplicadas.
    - columna (str): El nombre de la columna sobre la cual basar la búsqueda de duplicados.

    Returns:
    - pd.DataFrame o str: Si hay filas duplicadas, devuelve un DataFrame ordenado que contiene esas filas.
      Si no hay duplicados, devuelve el mensaje "No hay duplicados".
    """
   
    # Se filtran las filas duplicadas
    duplicacion_de_filas = df[df.duplicated(subset=columna, keep=False)]
    if duplicacion_de_filas.empty:
        return "No hay duplicados"
    
    # Se ordenan dichas filas y se comparan
    filas_duplicadas_ordenadas = duplicacion_de_filas.sort_values(by=columna)
    return filas_duplicadas_ordenadas



# Reemplazar valores nulos o vacío de la columna "price"

def reemplazo_nulos_y_strings(df, columna):
    """
    Esta función toma un DataFrame y un nombre de columna, y reemplaza los valores nulos
    o las cadenas en esa columna con 0.0 y convierte la columna a tipo float.
    En este caso, pd.to_numeric con downcast='float' intentará convertir todos los valores a tipo flotante. 
    Cualquier valor que no pueda ser convertido a flotante será reemplazado por NaN.
    Parameters:
    - df: DataFrame
        El DataFrame en el que se realizarán los reemplazos.
    - columna: str
        El nombre de la columna en la que se realizarán los reemplazos.

    Returns:
    - DataFrame
        El DataFrame modificado.
    """
    # Reemplaza los valores nulos con 0.0 en la columna especificada
    df[columna] = df[columna].fillna(0.0)

    # Reemplaza las cadenas con 0.0 en la columna especificada
    df[columna] = pd.to_numeric(df[columna], errors='coerce', downcast='float') 

    return df



def convertir_fecha(fecha):
    """
    Convierte una cadena de fecha en un formato específico a un objeto de fecha de Pandas.

    Parameters:
    - fecha: La cadena de fecha que se va a convertir.

    Returns:
    - str: Si la conversión es exitosa, devuelve la fecha en formato 'YYYY-MM-DD'.
      Si la cadena de fecha es NaN, devuelve 'Valor NaN'.
      Si la cadena de fecha no sigue el formato esperado, devuelve 'Formato inválido'.
      Si la conversión genera un error, devuelve 'Fecha inválida'.
    """
    try:
        # Verificar si la cadena de fecha no es NaN
        if pd.notna(fecha):
            # Buscar un patrón específico en la cadena de fecha
            match = re.search(r'(\w+\s\d{1,2},\s\d{4})', str(fecha))
            if match:
                # Extraer la fecha capturada del grupo 1 (lo que está dentro de los paréntesis)
                fecha_str = match.group(1)
                try:
                    # Convertir la cadena de fecha a un objeto de fecha de Pandas
                    fecha_dt = pd.to_datetime(fecha_str)
                    # Devolver la fecha formateada en el formato deseado
                    return fecha_dt.strftime('%Y-%m-%d')
                except ValueError:
                    return 'Fecha inválida'
            else:
                return 'Formato inválido'
        else:
            return 'Valor NaN'  
    except ValueError:
        return 'Error'
    



"""Para el análisis de sentimiento utilizaremos Textblob"""


def analisis_de_sentimiento(review):
    """
    Analiza el sentimiento de un texto utilizando la librería TextBlob.

    Parameters:
    - review (str): El texto del cual se quiere analizar el sentimiento.

    Returns:
    - int: 0 si el sentimiento es negativo, 1 si es neutral, 2 si es positivo.
      Si el argumento 'review' es None, devuelve 1 por defecto.
    """

    if review is None:
        return 1
    polarity =TextBlob(review).sentiment.polarity 
    if polarity < 0:
        return 0
    elif polarity > 0:
        return 2
    else:
        return 1
    


def analizar_ejemplos(ejemplos):
    """
    Analiza la polaridad promedio del sentimiento en una lista de ejemplos utilizando la librería TextBlob.

    Parameters:
    - ejemplos (list): Una lista de cadenas de texto que se desea analizar.

    Returns:
    - float: La polaridad promedio del sentimiento en los ejemplos.
      Si la lista de ejemplos está vacía, devuelve 0 asumiendo una polaridad neutra.
    """

    total_polaridad = 0

    for ejemplo in ejemplos:
        analysis = TextBlob(ejemplo)
        total_polaridad += analysis.sentiment.polarity

    if ejemplos:
        return total_polaridad / len(ejemplos)
    else:
        return 0  # Si no hay ejemplos, asumir una polaridad neutra




def ejemplos_review_por_sentimiento(reviews, sentiments):
    '''
    Imprime ejemplos de reviews para cada categoría de análisis de sentimiento.

    Esta función recibe dos listas paralelas, `reviews` que contiene los textos de las reviews
    y `sentiments` que contiene los valores de sentimiento correspondientes a cada review.
    
    Parameters:
        reviews (list): Una lista de strings que representan los textos de las reviews.
        sentiments (list): Una lista de enteros que representan los valores de sentimiento
                          asociados a cada review (0, 1, o 2).

    Returns:
        None: La función imprime los ejemplos de reviews para cada categoría de sentimiento.
    '''
    for sentiment_value in range(3):
        print(f"Para la categoría de análisis de sentimiento {sentiment_value} se tienen estos ejemplos de reviews:")
        sentiment_reviews = [review for review, sentiment in zip(reviews, sentiments) if sentiment == sentiment_value]
        
        for i, review in enumerate(sentiment_reviews[:3], start=1):
            print(f"Review {i}: {review}")
        
        print("\n")


def calcular_polaridad_promedio(ejemplos):
    """
    Calcula la polaridad promedio para una lista de ejemplos.

    Parámetros:
    - ejemplos: Una lista de cadenas de texto que representan ejemplos de sentimiento.

    Retorna:
    - Un valor entre -1 y 1 representando la polaridad promedio del sentimiento.
    """
    total_polaridad = 0

    for ejemplo in ejemplos:
        analysis = TextBlob(str(ejemplo))
        total_polaridad += analysis.sentiment.polarity

    if ejemplos:
        return total_polaridad / len(ejemplos)
    else:
        return 0  # Si no hay ejemplos, asumir una polaridad neutra


def obtener_anio_release(fecha):
    '''
    Extrae el año de una fecha en formato 'yyyy-mm-dd' y maneja valores nulos.

    Esta función toma como entrada una fecha en formato 'yyyy-mm-dd' y devuelve el año de la fecha si
    el dato es válido. Si la fecha es nula o inconsistente, devuelve None.

    Parameters:
        fecha (str or float or None): La fecha en formato 'yyyy-mm-dd'.

    Returns:
        int or None: El año de la fecha como un entero si es válido, None si es nula o el formato es incorrecto.
    '''
    if pd.notna(fecha):
        if re.match(r'^\d{4}-\d{2}-\d{2}$', fecha):
            return int(fecha.split('-')[0])
    return None

def extraer_anio(date_str):
    try:
        parts = date_str.split()
        year = int(parts[-1][:-1])  
        return year
    except (ValueError, IndexError):
        return 'sin fecha'
