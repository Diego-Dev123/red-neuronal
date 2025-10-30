#importamos librerias necesarias para el funcnionamiento 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

#importamos el excel de la carpeta daros

data_path='data/california_housing_test.xlsx'
df=pd.read_excel(data_path)
print("iformacion del data")
print(df.info())
print("\n 5 filas las primeras:")
print(df.head())
#separamos el 80 y 20
# Se divide el dataset en 80% q es pa el  entrenamiento y 20% pa la prueba
# random_state=123 asegura que la división sea reproducible
train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
print(f"\nTamaño entrenamiento: {len(train_df)}")
print(f"Tamaño prueba: {len(test_df)}")

#crear un pipelane

target='median_house_value' #es la variable la cual queremos predecir

#separamos las caracteristicas del est y train para el eje x y y

X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_test = test_df.drop(columns=[target])
y_test = test_df[target]

#despues de separar creamos el pipelane
# Crear pipeline con standardScaler normaliza los datos: media=0, desviación estándar=1
# Esto es importante porque las redes neuronales son sensibles a la escala
pipeline = Pipeline([
    ("scaler", StandardScaler())
])
#aqui es donde nosotros aplicamos la normalizacion a los datos
X_train_scaled = pipeline.fit_transform(X_train)# fit_transform: calcula parámetros de normalización y transforma X_train
X_test_scaled = pipeline.transform(X_test)# transform: aplica los mismos parámetros a X_test sin recalcular

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1)).ravel()

print("\nDatos normalizados.")

# Justificación de StandardScaler:
# La normalización acelera la convergencia del algoritmo
# Previene que variables con mayor escala dominen el aprendizaje y segun documentacion de redes pues esto hace mas eficiente
# Es práctica estándar en redes neuronales osea q si o si debe ir


# 4. ENTRENAR 5 REDES CON SOLVER 'adam' Y ACTIVACIÓN 'logistic'
# Definir las arquitecturas de red a probar 
# Cada tupla representa las capas ocultas neurona_capa1, neurona_capa2 y asi sucesivamente
arquitecturas = [
    (50,),#1 capa con 50 neuronas
    (100,),#1 capa con 100 neuronas
    (100, 50),# 2 capas 100  50
    (100, 100, 50),#3 capas 100 100 50
    (200, 100, 50) #3 capas 200 100 50
]

resultados = []
for hidden in arquitecturas:
    # Crear modelo de red neuronal
    # solver='adam' optimizador basado en gradiente estocástico
    # activation='logistic': función de activación sigmoide
    # max_iter=500: número máximo de iteraciones para el entrenamiento
    model = MLPRegressor( #aqui es donde se crea el modelo de la red neuronal
        hidden_layer_sizes=hidden,
        solver='adam',
        activation='logistic',
        max_iter=2000,
        random_state=123
    )

    #empezamos con el entrenamiento
    model.fit(X_train_scaled, y_train_scaled)
    #predicciones
    preds_scaled = model.predict(X_test_scaled)
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
    #calcular metricas
    # R2 coeficiente de determinación en resumen esto es el mas alto es mejor
    r2 = r2_score(y_test, preds)
    # RMSE raíz del error cuadrático medio aqui es donde ser calcular el error
    rmse = mean_squared_error(y_test, preds) ** 0.5
    accuracy = (1 - (rmse / y_test.mean())) * 100

    resultados.append(["adam_logistic", hidden, accuracy, rmse]) #guardar resiltado
    print(f"Capas {hidden} -> Accuracy={accuracy:.2f}%, RMSE={rmse:.2f}") #imprimir resultados

resultados_df = pd.DataFrame(resultados, columns=["Tipo", "Capas ocultas", "Accuracy (%)", "RMSE"])
print("\n=== TABLA ACCURACY: adam + logistic ===")
print(resultados_df.sort_values("Accuracy (%)", ascending=False))

#si queremos saber cual es el mejor pues imprimimos el mayor r2
mejor = resultados_df.loc[resultados_df["Accuracy (%)"].idxmax()]
print("\n=== MEJOR MODELO (adam + logistic) ===")
print(mejor)

def auto_pipeline(df, target, arquitecturas, solvers_activaciones):
    """
    Parámetros:
    df: DataFrame con los datos
    target: nombre de la columna objetivo
    arquitecturas: lista de tuplas con capas ocultas
    solvers_activaciones: lista de tuplas (solver, activation)
    
    Retorna:
    DataFrame con resultados de r2"""

    # Separar características y objetivo
    X = df.drop(columns=[target]) #drop elimina la columna target
    y = df[target]
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123 #test size 20% y random state para la produccion
    )
    
    # Normalizar datos
    scaler = StandardScaler() #StandardScaler para normalizar 
    X_train_s = scaler.fit_transform(X_train) #scaler.fit_transform para entrenar y transformar
    X_test_s = scaler.transform(X_test) #scaler.transform para transformar los datos de prueba

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
    y_test_s = scaler_y.transform(y_test.values.reshape(-1,1)).ravel()

    resultados_auto = [] #lista donde se van a guardar los resultad

    #probamos todas las combinaciones de arquitecturas y solvers/activaciones
    for solver, activation in solvers_activaciones:
        for hidden in arquitecturas:
            model = MLPRegressor(
                hidden_layer_sizes=hidden,
                solver=solver,
                activation=activation,
                max_iter=2000,
                random_state=123
            )
            model.fit(X_train_s, y_train_s) #model.fit para entrenar el modelo
            
            # Evaluar modelo
            preds_s = model.predict(X_test_s)#predecir 
            preds = scaler_y.inverse_transform(preds_s.reshape(-1,1)).ravel()
            r2 = r2_score(y_test, preds) #calcula el r2
            rmse = mean_squared_error(y_test, preds) ** 0.5 #el error
            accuracy = (1 - (rmse / y_test.mean())) * 100
            
            # Guardar resultados
            resultados_auto.append([solver, activation, hidden, accuracy, rmse])
    #retornar resultados
    return pd.DataFrame(
        resultados_auto,
        columns=["Solver", "Activation", "Capas ocultas", "Accuracy (%)", "RMSE"]
    )

print("\n##### DEMO DE AUTO_PIPELINE #######")
print("Uso: auto_pipeline(df, 'target', [(50,), (100,)], [('adam','relu')])")
solvers_activaciones = [("adam", "logistic"), ("lbfgs", "relu")]
df_result = auto_pipeline(df, "median_house_value", arquitecturas, solvers_activaciones)

print("\n##### TABLA ACCURACY: adam + logistic #####")
adam_log = df_result[df_result["Solver"] == "adam"].sort_values("Accuracy (%)", ascending=False)
print(adam_log)

print("\n##### TABLA ACCURACY: lbfgs + relu #####")
lbfgs_relu = df_result[df_result["Solver"] == "lbfgs"].sort_values("Accuracy (%)", ascending=False)
print(lbfgs_relu)

print("\n####### MEJOR MODELO GLOBAL ########")
mejor_global = df_result.loc[df_result["Accuracy (%)"].idxmax()]
print(mejor_global)

# tener en cuenta q:
#• Accuracy = (1 - RMSE/promedio_real) * 100
#• max_iter=2000 para mejor convergencia (lbfgs necesita más iteraciones)
#• lbfgs es más lento pero preciso, adam es más rápido