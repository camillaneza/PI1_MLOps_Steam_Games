from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from Jupyter import Funciones_API
from typing import List, Dict
import importlib
importlib.reload(Funciones_API)
import fastparquet


"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
"""

df_combinado = pd.read_parquet('Jupyter\df_combinado.parquet')
df_combinado2 = pd.read_parquet('Jupyter\df_combinado2.parquet')


app = FastAPI(title= 'Proyecto Integrador 1',
              description= 'Machine Learning Operations (MLOps)',
              version= '1.0.1', debug=True)


@app.get("/")
async def read_root():
    return  {"Proyecto Integrador 1: Machine Learning Operations (MLOps)"}

@app.get('/about/')
async def about():
    return 'Proyecto Integrador 1: Machine Learning Operations'

"""

@app.get('/PlayTimeGenre/{genero}')
async def PlayTimeGenre(genero: str):
    try:
        df_1 = pd.read_parquet('Jupyter/df_combinado2.parquet')
        resultado = Funciones_API.Play_Time_Genre(df_1, genero)
        return resultado
    except HTTPException as e:
        return {"error": e.detail}




@app.get('/UserForGenre/{genero}')
async def UserForGenre(genero: str):
    try:
        df_2 = pd.read_parquet('Jupyter/df_combinado2.parquet')
        resultado = Funciones_API.user_for_genre(df_2, genero)

        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get('/UsersRecommend/{año}')
async def UsersRecommend(año: int):
    try:
        ruta_parquet_df_combinado = r'..\Jupyter\udf_combinado.parquet'
        df_3 = pd.read_parquet(ruta_parquet_df_combinado)
        resultado = Funciones_API.Users_Recommend(df_3, año)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/UsersNotRecommend/{año}')
async def UsersNotRecommend(año: int):
    try:
        df_4 = pd.read_parquet('Jupyter/df_combinado.parquet')
        resultado = Funciones_API.users_not_recommend(df_4, año)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/sentiment_analysis/{año}')
def sentiment_analysis(año):
    try:
        df_5 = pd.read_parquet('Jupyter\df_combinado.parquet')
        resultado = Funciones_API.analisis_sentimiento(df_5, año)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""