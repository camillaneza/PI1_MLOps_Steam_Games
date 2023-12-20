from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from Jupyter import Funciones_API
from typing import List, Dict
import importlib
importlib.reload(Funciones_API)
from fastparquet import ParquetFile
import os




ruta_archivo_combinado = os.path.join('Jupyter', 'df_combinado.parquet')
ruta_archivo_combinado2 = os.path.join('Jupyter', 'df_combinado2.parquet')

parquet_file = ParquetFile(ruta_archivo_combinado)
parquet_file2 = ParquetFile(ruta_archivo_combinado2)

df_combinado = parquet_file.to_pandas()
df_combinado2 = parquet_file2.to_pandas()


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