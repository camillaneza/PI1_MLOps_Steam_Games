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


parquet_file_path2 = "Jupyter/df_combinado2_gzip.parquet"

try:
    sample_percent = 5


    # Lee una muestra del archivo Parquet con pyarrow
    parquet_file2 = pq.ParquetFile(parquet_file_path2)
    total_rows2 = parquet_file2.metadata.num_rows
    sample_rows2 = int(total_rows2 * (sample_percent / 100.0))
    df_combinado_muestra2 = parquet_file2.read_row_groups(row_groups=[0]).to_pandas().head(sample_rows2//50)

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




@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero: str):
    '''
    Datos:
    - genero (str): Género para el cual se busca el año con más horas jugadas.

    Funcionalidad:
    - Devuelve el año con más horas jugadas para el género especificado.

    Return:
    - Dict: {"Año de lanzamiento con más horas jugadas para Género X": int}
    '''
    try:
        genero_filtrado = df_combinado_muestra2[df_combinado_muestra2['genres'].apply(lambda x: genero in x)]

        if genero_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No hay datos para el género {genero}")

        genero_filtrado['playtime_forever'] = genero_filtrado['playtime_forever'] / 60

        max_hours_year = genero_filtrado.groupby('release_date')['playtime_forever'].sum().idxmax()

        return {"Año de lanzamiento con más horas jugadas para el Género " + genero: int(max_hours_year)}

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