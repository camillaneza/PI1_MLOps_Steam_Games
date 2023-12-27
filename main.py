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
parquet_file_path2 = "Jupyter/df_combinado2_gzip.parquet"

try:
    sample_percent = 5

    # Lee una muestra del archivo Parquet con pyarrow
    parquet_file1 = pq.ParquetFile(parquet_file_path1)
    total_rows1 = parquet_file1.metadata.num_rows
    sample_rows1 = int(total_rows1 * (sample_percent / 100.0))
    df_combinado_muestra1 = parquet_file1.read_row_groups(row_groups=[0]).to_pandas().head(sample_rows1)

    # Lee una muestra del archivo Parquet con pyarrow
    parquet_file2 = pq.ParquetFile(parquet_file_path2)
    total_rows2 = parquet_file2.metadata.num_rows
    sample_rows2 = int(total_rows2 * (sample_percent / 100.0))
    df_combinado_muestra2 = parquet_file2.read_row_groups(row_groups=[0]).to_pandas().head(sample_rows2//5)

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
        
        condition = df_combinado_muestra2['genres'].apply(lambda x: genero in x)
        juegos_genero = df_combinado_muestra2[condition]

       
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

@app.get('/RecomendacionJuego/{id_producto}')
async def recomendacion_juego(id_producto: int = Path(..., description="ID del juego para obtener recomendaciones")):
    """
    Endpoint para obtener una lista de juegos recomendados similares a un juego dado.

    Parámetros:
    - id_juego (int): ID del juego para el cual se desean obtener recomendaciones.

    Respuestas:
    - 200 OK: Retorna una lista con 5 juegos recomendados similares al juego ingresado.
    - 404 Not Found: Si no se encuentra el juego con el ID especificado.
    - 500 Internal Server Error: En caso de cualquier otro error, proporciona detalles de la excepción.

    Ejemplo de Uso:
    - /RecomendarJuego/123

    Ejemplo de Respuesta Exitosa:
    [
        {"id": 456, "nombre": "Juego A"},
        {"id": 789, "nombre": "Juego B"},
        {"id": 101, "nombre": "Juego C"},
        {"id": 202, "nombre": "Juego D"},
        {"id": 303, "nombre": "Juego E"}
    ]
    """
    try:
        porcentaje_muestra = 50
        total_registros = len(df_combinado_muestra2)

        num_registros = int(total_registros * (porcentaje_muestra / 100))

        df_subset = df_combinado_muestra2.sample(n=num_registros, random_state=42).reset_index(drop=True)

        num_recommendations = 5

        juego_seleccionado = df_subset[df_subset['item_id'] == id_producto]

        if juego_seleccionado.empty:
            raise HTTPException(status_code=404, detail=f"No se encontró el juego con ID {id_producto}")

        title_game_and_genres = ' '.join(juego_seleccionado['title'].fillna('').astype(str) + ' ' + juego_seleccionado['genres'].fillna('').astype(str))
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_subset['title'].fillna('').astype(str) + ' ' + df_subset['genres'].fillna('').astype(str))

        juego_tfidf = tfidf_vectorizer.transform([title_game_and_genres])
        similarity_scores = cosine_similarity(juego_tfidf, tfidf_matrix)

        if similarity_scores is not None:
            similar_games_indices = similarity_scores[0].argsort()[::-1]

            
            recommended_games = df_subset.loc[similar_games_indices[1:]]
            recommended_games = recommended_games[~recommended_games['item_id'].isin([id_producto])].drop_duplicates(subset='title')

            recommendations_list = recommended_games.head(num_recommendations)['title'].tolist()

            if len(recommendations_list) < num_recommendations:
                message = f"Se encontraron {len(recommendations_list)} recomendaciones para este ID."
                recommendations_list += [None] * (num_recommendations - len(recommendations_list))
            else:
                message = None

            return {"recomendaciones": recommendations_list, "message": message}
        else:
            return {"message": "No se encontraron juegos similares."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}") from e





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