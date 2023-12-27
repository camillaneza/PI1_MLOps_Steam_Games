from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import HTMLResponse
import pandas as pd
from typing import List, Dict
import os
import pyarrow.parquet as pq
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import gzip


parquet_file_path1 = "Jupyter/df_combinado_gzip.parquet"


try:
    sample_percent = 5

    # Lee una muestra del archivo Parquet con pyarrow
    parquet_file1 = pq.ParquetFile(parquet_file_path1)
    total_rows1 = parquet_file1.metadata.num_rows
    sample_rows1 = int(total_rows1 * (sample_percent / 100.0))
    df_combinado_muestra1 = parquet_file1.read_row_groups(row_groups=[0]).to_pandas().head(sample_rows1)

except FileNotFoundError:
    # Si alguno de los archivos no se encuentra, maneja la excepción
    raise HTTPException(status_code=500, detail="Error al cargar el archivo de datos Parquet")
except Exception as e:
    # Si ocurre cualquier otra excepción, imprime información detallada
    print(f"Ocurrió una excepción: {str(e)}")
    raise HTTPException(status_code=500, detail="Error al cargar el archivo de datos Parquet")

app = FastAPI(title= 'Proyecto Integrador 1',
              description= 'Machine Learning Operations (MLOps), por Camila Fernández Llaneza',
              version= '1.0.1', debug=True)



@app.get('/UsersRecommend/{anio}')
def UsersRecommend(anio: int):
    '''
    Datos:
    - anio (int): Año para el cual se busca el top 3 de juegos más recomendados.

    Funcionalidad:
    - Devuelve el top 3 de juegos más recomendados por usuarios para el año dado.

    Return:
    - List: [{"Puesto 1": str}, {"Puesto 2": str}, {"Puesto 3": str}]
    '''
    try:
        filtered_df = df_combinado_muestra1[
        (df_combinado_muestra1["reviews_posted"] == anio) &
        (df_combinado_muestra1["reviews_recommend"] == True) &
        (df_combinado_muestra1["sentiment_analysis"]>=1)
        ]
        recommend_counts = filtered_df.groupby("title")["title"].count().reset_index(name="count").sort_values(by="count", ascending=False).head(3)
        top_3_dict = {f"Puesto {i+1}": juego for i, juego in enumerate(recommend_counts['title'])}
        return top_3_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al obtener los juegos mas recomendados.")

@app.get('/UsersNotRecommend/{anio}')
def UsersNotRecommend(anio: int):
    '''
    Datos:
    - anio (int): Año para el cual se busca el top 3 de juegos menos recomendados.

    Funcionalidad:
    - Devuelve el top 3 de juegos menos recomendados por usuarios para el año dado.

    Return:
    - List: [{"Puesto 1": str}, {"Puesto 2": str}, {"Puesto 3": str}]
    '''
    try:
        filtered_df = df_combinado_muestra1[
        (df_combinado_muestra1["reviews_posted"] == anio) &
        (df_combinado_muestra1["reviews_recommend"] == False) &
        (df_combinado_muestra1["sentiment_analysis"]==0)
        ]
        not_recommend_counts = filtered_df.groupby("title")["title"].count().reset_index(name="count").sort_values(by="count", ascending=False).head(3)
        top_3_dict = {f"Puesto {i+1}": juego for i, juego in enumerate(not_recommend_counts['title'])}
        return top_3_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al obtener los juegos menos recomendados.")
    
@app.get('/sentiment_analysis/{anio}')
def sentiment_analysis(anio: int):

    '''
    Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

    Args:
        año (int): Año para el cual se busca el análisis de sentimiento.

    Returns:
        dict: Diccionario con la cantidad de reseñas por sentimiento.
    '''
  
    try:    
        filtered_df = df_combinado_muestra1[df_combinado_muestra1["release_date"] == anio]

        
        sentiment_counts = filtered_df["sentiment_analysis"].value_counts()

        
        sentiment_mapping = {2: "Positive", 1: "Neutral", 0: "Negative"}
        sentiment_counts_mapped = {sentiment_mapping[key]: value for key, value in sentiment_counts.items()}

        return sentiment_counts_mapped
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=404, detail=f"No hay datos para el año {anio}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






# Inicio
    
@app.get("/", response_class=HTMLResponse, tags=["Home"])
async def presentacion():
    return '''
        <html>
            <head>
                <title>API Steam</title>
                <style>
                    body {
                        color: black; 
                        background-color: white; 
                        font-family: Arial, sans-serif;
                        padding: 20px;
                    }
                    h1 {
                        color: #333;
                        text-align: center;
                    }
                    p {
                        color: #666;
                        text-align: center;
                        font-size: 18px;
                        margin-top: 20px;
                    }
                    footer {
                        text-align: center;
                    }
                </style>
            </head>
            <body>
                <h1>Proyecto Individual N° 1: MLOps Steam</h1>
                <p>Esta es una API para consultas de la plataforma Steam.</p>
                <p>Escriba <span style="background-color: lightgray;">/docs</span> a continuación de la URL actual para ingresar.</p>
                
            </body>
        </html>
    '''


# Funciones

@app.get(path='/PlayTimeGenre/{genero}', tags=["Funciones Generales"])
def play_time_genre(genero: str = Path(..., description="Devuelve el año con más horas jugadas para el género especificado (Ingresar la primer letra en Mayúscula)")):
    return PlayTimeGenre(genero)

@app.get(path='/UserForGenre/{genero}', tags=["Funciones Generales"])
def user_for_genre(genero: str = Path(..., description="Devuelve el usuario que acumula más horas jugadas para el género especificado (Ingresar la primer letra en Mayúscula)")):
    return UserForGenre(genero)


@app.get("/UsersRecommend/{anio}", tags=["Funciones Generales"])
def users_recommend(anio: int = Path(..., description="Devuelve el top 3 de juegos más recomendados para el año especificado")):
        try:
            result = UsersRecommend(anio)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get(path='/UsersNotRecommend/{anio}', tags=["Funciones Generales"])
def users_not_recommend(anio: int = Path(..., description="Devuelve el top 3 de juegos menos recomendados para el año especificado")):
    return UsersNotRecommend(anio)

@app.get(path='/sentiment_analysis/{anio}', tags=["Funciones Generales"])
def sentiment_analysis(anio: int = Path(..., description="Devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentran categorizados con un análisis de sentimiento en el año especificado")):
    return sentiment_analysis(anio)

@app.get(path='/RecomendacionJuego/{id_producto}', tags=["Sistema de Recomendación: Item-Item"])
async def recomendacion_juego(id_producto: int = Path(..., description= "Devuelve una lista con 5 juegos recomendados similares al ingresado")):
    return recomendacion_juego(id_producto)