import pandas as pd
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import HTMLResponse
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict
import fastparquet
import dask.dataframe as dd
import dask



def Play_Time_Genre(df, genero):
    try:
        genero_filtrado = df[df['genres'].str.contains(genero, case=False, na=False)]
        if genero_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No hay datos de horas jugadas para el género {genero}")

        df['playtime_forever'] = df['playtime_forever'] / 60

        año_mayor_horas = genero_filtrado.groupby('release_date')['playtime_forever'].sum().idxmax()
        año_mayor_horas = str(año_mayor_horas)

        return {"Año de lanzamiento con más horas jugadas para el Género " + genero: año_mayor_horas}
    
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Error al cargar los archivos de datos")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def user_for_genre(df, genero):
    """
    Encuentra el usuario con más horas jugadas para un género específico.

    Parameters:
    - df_combinado2_parquet (DataFrame): DataFrame que contiene datos de juegos, usuarios y horas jugadas.
    - genero (str): Género para el cual se desea encontrar el usuario con más horas jugadas.

    Returns:
    - dict: Un diccionario que contiene la información sobre el usuario con más horas jugadas para el género especificado.
      Ejemplo de estructura:
      {
        "Usuario con más horas jugadas para Género [genero]": [usuario_id],
        "Horas jugadas": [{"Año": [año], "Horas": [horas]}]
      }

    Raises:
    - HTTPException(404): Si no hay datos disponibles para el género proporcionado.
    - HTTPException(500): Si ocurre algún error durante la ejecución de la función.
    """
    try:
        genero_filtrado = df[df['genres'].str.contains(genero, case=False, na=False)]
        if genero_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No hay datos de horas jugadas para el género {genero}")

        df['playtime_forever'] = df['playtime_forever'] / 60
        # Agrupar por usuario y año, sumar las horas jugadas
        agrup_por_usuario_y_año = genero_filtrado.groupby(['user_id', 'release_date'])['playtime_forever'].sum().reset_index()

        # Encontrar el usuario con más horas jugadas
        usuario_max_horas = agrup_por_usuario_y_año.loc[agrup_por_usuario_y_año['playtime_forever'].idxmax(), 'user_id']

        # Filtrar las horas jugadas del usuario con más horas
        horas_usuario_max = agrup_por_usuario_y_año[agrup_por_usuario_y_año['user_id'] == usuario_max_horas]

        # Formatear el resultado
        resultado = {
            "Usuario con más horas jugadas para Género {}".format(genero): usuario_max_horas,
            "Horas jugadas": [{"Año": año, "Horas": horas} for año, horas in zip(horas_usuario_max['release_date'], horas_usuario_max['playtime_forever'])]
        }

        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
def user_recommend(df: pd.DataFrame, año: int, num_juegos: int = 3) -> List[dict]:
    
    try:
        # Filtrar el DataFrame por el año especificado y por juegos recomendados
        juegos_recomendados = df[(df['release_date'] == año) & (df['reviews_recommend'] == True) & (df['sentiment_analysis'] >= 1)]
        if juegos_recomendados.empty:
            raise Exception(f"No hay datos para el año {año} con los filtros especificados.")

        # Contar las recomendaciones para cada juego
        contar_recomendaciones = juegos_recomendados.groupby('title')['reviews_recommend'].count().reset_index()
        if contar_recomendaciones.empty:
            raise Exception(f"No hay juegos recomendados para el año {año} con los filtros especificados.")

        # Ordenar los juegos por la cantidad de recomendaciones en orden descendente
        top_juegos_recomendados = contar_recomendaciones.nlargest(num_juegos, 'reviews_recommend')

        # Formatear el resultado en el formato requerido
        resultado = [{"Puesto {}".format(pos + 1): juego} for pos, juego in enumerate(top_juegos_recomendados['title'])]

        return resultado
    except Exception as e:
        raise ValueError(f"Error al obtener juegos recomendados: {str(e)}")
"""


def Users_Recommend(df, año):
    try:
        filtro = (df['reviews_posted'] == año) & (df['reviews_recommend'] == True) & (df['sentiment_analysis'] >= 1)
        reviews = df[filtro]
        juegos_recomendados = reviews.groupby(df['title']).count().reset_index()
        juegos_recomendados = juegos_recomendados.sort_values(by='count', ascending=False).head(3)
        juegos_recomendados = juegos_recomendados.reset_index(drop=True)
        top_3 = {f"Puesto {i+1}": juego for i, juego in enumerate(juegos_recomendados['title'])}
        return top_3
    except Exception as e:
        raise HTTPException(status_code=500, detail='Error al cargar el archivo de datos')

 

"""

def obtener_nombre_juego2(item_id: int, df: pd.DataFrame) -> str:
    
    Función para obtener el nombre de un juego dado su identificador.
    
    juego = df[df['item_id'] == item_id].iloc[0]
    return juego['title']

def users_not_recommend(df: pd.DataFrame, año: int, num_juegos: int = 3) -> List[dict]:
  
    try:
        # Filtrar el DataFrame por el año especificado y por juegos no recomendados
        juegos_no_recomendados = df[(df['release_anio'] == año) & (df['recommend'] == False)]

        # Contar las no recomendaciones para cada juego
        count_no_recomendaciones = juegos_no_recomendados.groupby('item_id')['recommend'].count().reset_index()

        # Ordenar los juegos por la cantidad de no recomendaciones en orden descendente
        top_juegos_no_recomendados = count_no_recomendaciones.sort_values(by='recommend', ascending=False).head(num_juegos)

        # Obtener los nombres de los juegos correspondientes a los identificadores
        top_juegos_nombres2 = [obtener_nombre_juego2(item_id, df) for item_id in top_juegos_no_recomendados['item_id']]

        # Formatear el resultado en el formato requerido
        resultado = [{"Puesto {}".format(pos + 1): juego} for pos, juego in enumerate(top_juegos_nombres2)]

        return resultado
    except Exception as e:
        raise ValueError(f"Error al obtener juegos menos recomendados: {str(e)}")

"""
def analisis_sentimiento(df, año):
    try:
        juegos_año = df[df['release_date'] == año]
        if juegos_año.empty:
            raise Exception(f"No hay datos para el año {año} con los filtros especificados.")
        resultados = {
            'Negative': len(df[df['sentiment_analysis'] == 0]),
            'Neutral': len(df[df['sentiment_analysis'] == 1]),
            'Positive': len(df[df['sentiment_analysis'] == 2])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



"""def sentiment_analysis(anio):
    positive = 0
    negative = 0
    neutral = 0
    filtro = (df_reviews_year["reviews_posted"] == anio)
    reviews = df_reviews_year[filtro]
    
    for i in  reviews["sentiment_analysis"]:
        if i == 2:
            positive +=1
        elif i == 1:
            neutral +=1
        elif i == 0:
            negative +=1
    resultado = [
    {"Negativas: ",negative," Positivas: ",positive," Neutrales: ",neutral}]
    return resultado"""