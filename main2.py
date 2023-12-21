from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from Jupyter import Funciones_API
from typing import List, Dict
import importlib
importlib.reload(Funciones_API)
import os

file_path = os.path.join("Jupyter", "df_combinado.parquet")
df_combinado = pd.read_parquet(file_path)

file_path = os.path.join("Jupyter", "df_combinado2.parquet")
df_combinado2 = pd.read_parquet(file_path)

app = FastAPI(title= 'Proyecto Integrador 1',
              description= 'Machine Learning Operations (MLOps)',
              version= '1.0.1', debug=True)


@app.get("/")
async def read_root():
    return  {"Proyecto Integrador 1: Machine Learning Operations (MLOps)"}

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
        genero_filtrado = df_combinado2[df_combinado2['genres'].apply(lambda x: genero in x)]

        if genero_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No hay datos para el género {genero}")

        genero_filtrado['playtime_forever'] = genero_filtrado['playtime_forever'] / 60

        max_hours_year = genero_filtrado.groupby('release_date')['playtime_forever'].sum().idxmax()

        return {"Año de lanzamiento con más horas jugadas para el Género " + genero: int(max_hours_year)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/UserForGenre/{genero}')
def UserForGenre(genero:str):
    '''
    Datos:
    - genero (str): Género para el cual se busca el usuario con más horas jugadas y la acumulación de horas por año.

    Funcionalidad:
    - Devuelve el usuario con más horas jugadas y una lista de la acumulación de horas jugadas por año para el género especificado.

    Return:
    - Dict: {"Usuario con más horas jugadas para Género X": List, "Horas jugadas": List}
    '''
    try:
        
        condition = df_combinado2['genres'].apply(lambda x: genero in x)
        juegos_genero = df_combinado2[condition]

       
        juegos_genero['playtime_forever'] = juegos_genero['playtime_forever'] / 60
        juegos_genero['release_date'] = pd.to_numeric(juegos_genero['release_date'], errors='coerce')
        juegos_genero = juegos_genero[juegos_genero['release_date'] >= 100]
        juegos_genero['Año'] = juegos_genero['release_date']

        horas_por_usuario = juegos_genero.groupby(['user_id', 'Año'])['playtime_forever'].sum().reset_index()
        if not horas_por_usuario.empty:
            usuario_max_horas = horas_por_usuario.groupby('user_id')['playtime_forever'].sum().idxmax()
            usuario_max_horas = horas_por_usuario[horas_por_usuario['user_id'] == usuario_max_horas]
        else:
            usuario_max_horas = None

        acumulacion_horas = horas_por_usuario.groupby(['Año'])['playtime_forever'].sum().reset_index()
        acumulacion_horas = acumulacion_horas.rename(columns={'Año': 'Año', 'playtime_forever': 'Horas'})

        resultado = {
            "Usuario con más horas jugadas para " + genero: {"user_id": usuario_max_horas.iloc[0]['user_id'], "Año": int(usuario_max_horas.iloc[0]['Año']), "playtime_forever": usuario_max_horas.iloc[0]['playtime_forever']},
            "Horas jugadas": [{"Año": int(row['Año']), "Horas": row['Horas']} for _, row in acumulacion_horas.iterrows()]
        }

        return resultado
        

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Error al cargar los archivos de datos")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        filtered_df = df_combinado[
        (df_combinado["reviews_posted"] == anio) &
        (df_combinado["reviews_recommend"] == True) &
        (df_combinado["sentiment_analysis"]>=1)
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
        filtered_df = df_combinado[
        (df_combinado["reviews_posted"] == anio) &
        (df_combinado["reviews_recommend"] == False) &
        (df_combinado["sentiment_analysis"]==0)
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
        filtered_df = df_combinado[df_combinado["release_date"] == anio]

        
        sentiment_counts = filtered_df["sentiment_analysis"].value_counts()

        
        sentiment_mapping = {2: "Positive", 1: "Neutral", 0: "Negative"}
        sentiment_counts_mapped = {sentiment_mapping[key]: value for key, value in sentiment_counts.items()}

        return sentiment_counts_mapped
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=404, detail=f"No hay datos para el año {anio}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))