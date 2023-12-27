<p align="center">
  <b style="font-size: 24px;"> PROYECTO INDIVIDUAL 1: MLOPS:rocket:</b>
</p>

<p align="center">
 <b style="font-size: 24px;"> PLATAFORMA STEAM GAMES:joystick:</b>
</p>

<p align="center">
  <i><u>INTRODUCCIÓN:</u></i>
</p>

En este proyecto se va a trabajar sobre la plataforam de juegos Steam, desarrollando un rol de Data Engineer para lograr tener un MVP (Minimum Viable Product). En base a 3 datasets iniciales, se va a desarrollar el correspondiente proceso de ETL (Extracción, Transformación y Carga) y de EDA (Análisis Exploratorio de Datos). Además, se va a realizar un  análisis de sentimiento con NLP, y un Modelo de aprendizaje automático, con un sistema de recomendación.

<p align="center">
  <i><u>ARCHIVOS INICIALES:</u></i>
</p>

Se tienen 3 datasets:
• australian_user_reviews.json: Conjunto de datos con id de usuarios y sus comentrios de los juegos, su recomendación o no, así como también la url del perfil de usuario y el id del juego.

• australian_users_items.json: Conjunto de datos con información de los juegos, y el tiempo acumulado de juego por cada usuario.

• output_steam_games.json: Conjunto de datos con títulos, géneros, id de los juegos, sus precios y características.

<p align="center">
  <i><u>PROCESO DE ETL, EDA y FEATURE ENGINEERING:</u></i>
</p>

Como se mencionó anteriormente, como primera medida se llevó a cabo un proceso de ETL (Extracción, Transformación y Carga), analizando el tipo de dato de cada columna de los distintos datasets, transformándolos cuando fuera necesario, eliminando duplicados, eliminando columnas con valores nulos, desañidando 2 columnas que estaban - justamente - añidadas. También se procedió a eliminar las columnas que no iban a ser de utilidad para el posterior análisis y funcionamiento de api.
Para la realización de la consigna de realizar un un análisis de sentimiento a los comentarios de los usuarios, se introdujo una nueva columna llamada 'sentiment_analysis', la cual sustituye a la columna que originalmente contenía los comentarios de los usuarios. Esta columna clasifica los sentimientos de los comentarios según la siguiente escala:  0 si el sentimiento es negativo, 1 si es neutral o si no hay un comentario asociado,  2 si el sentimiento es positivo. 
Dado que se pedía que se aplicara un análisis de sentimiento con NLP, se utilizó la biblioteca TextBlob, que clasifica la polaridad del texto como positiva, negativa o neutra. 
Se guardaron los datasets limpios en archivos parquet. Luego se procedió a la realización del EDA (Análisis Exploratorio de Datos), para identificar los datos necesarios para la posterior realización del modelo de recomendación. Se usaron las librerías Matplotlib y Seaborn para la visualización.
Todo lo antedicho se puede ver en los archivos:  
[ETL_Steam_Games](Jupyter/ETL_Steam_Games.ipynb)  
[ETL_user_items](Jupyter/ETL_user_items.ipynb)  
[ETL_users_reviews](Jupyter/ETL_users_reviews.ipynb)  
[Feature_Engineering_EDA](Jupyter/Feature_Engineering_EDA.ipynb)  


<p align="center">
  <i><u>API:</u></i>
</p>

El desarrollo de la API se realizó usando el framework FastAPI, generando las 5 funciones propuestas para las consultas:
• _PlayTimeGenre:_ Debe devolver año con mas horas jugadas para dicho género.

• _UserForGenre:_ Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

• _UsersRecommend:_ Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.

• _UsersNotRecommend:_ Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.

• _sentiment_analysis( año : int ):_ Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

Posteriormente, se realizó el Modelo de Recomendación Automático, utilizando el sistema de recomendación item-item:
• _recomendacion_juego( id de producto ):_ Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Lo antedicho se puede observar en el archivo [main](main.py)  

<p align="center">
  <i><u>DEPLOYMENT:</u></i>
</p>

Luego de verificar que la API funciona a nivel local, se procedió a usar Render para que la misma pueda ser consumida desde la web. Dado que el servicio gratuito de Render consta de poca memoria, se optó por un muestreo porcentual de los Dataframes pertinentes.