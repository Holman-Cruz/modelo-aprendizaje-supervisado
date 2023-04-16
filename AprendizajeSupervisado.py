# Importamos librerias
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

# Cargamos el dataset
df = pd.read_csv('transmilenio.csv')

# Dividimos los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[['Distancia', 'Paradas']], df['Tiempo'], test_size=0.2, random_state=42)


# Creamos el modelo de regresión lineal
model = LinearRegression()
model.fit(df[['Distancia', 'Paradas']], df['Tiempo'])

# Ajustamos el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Se realizan las predicciones sobre los datos de prueba
y_pred = model.predict(X_test)

# Se evalua el rendimiento del modelo
score = model.score(X_test, y_test)
print(f'R2 del modelo: {score:.2f}')

# Creamos la gráfica de dispersión con los datos de entrenamiento
plt.scatter(X_train['Distancia'], y_train, label='Datos de entrenamiento')
plt.plot(X_train['Distancia'], model.predict(X_train), color='red', label='Línea de regresión lineal')

# Agregamos la leyenda y títulos 
plt.legend()
plt.title('Tiempo de recorrido de ruta Transmilenio')
plt.xlabel('Distancia (km)')
plt.ylabel('Tiempo (min)')

# Mostramos la gráfica
plt.show()


# Definimos la función para calcular el tiempo
def calcular_tiempo():

    distancia = float(entry_distancia.get())
    paradas = int(entry_paradas.get())

    input_array = np.array([distancia, paradas]).reshape(1, 2)

    input_df = pd.DataFrame(input_array, columns=['Distancia', 'Paradas'])

    tiempo = model.predict(input_df)

    label_resultado.config(text=f'Tiempo estimado: {tiempo[0]:.2f} min')


# Creamos la ventana emergente
ventana = tk.Tk()

# Incluimos los elementos a la ventana
label_distancia = tk.Label(ventana, text='Distancia (km):')
entry_distancia = tk.Entry(ventana)
label_paradas = tk.Label(ventana, text='Número de paradas:')
entry_paradas = tk.Entry(ventana)
button_calcular = tk.Button(ventana, text='Calcular tiempo', command=calcular_tiempo)
label_resultado = tk.Label(ventana, text='')

label_distancia.grid(row=0, column=0, padx=20, pady=10)
entry_distancia.grid(row=0, column=1, padx=20, pady=10)
label_paradas.grid(row=1, column=0, padx=20, pady=10)
entry_paradas.grid(row=1, column=1, padx=20, pady=10)
button_calcular.grid(row=2, column=0, columnspan=2, padx=20, pady=10)
label_resultado.grid(row=3, column=0, columnspan=2, padx=20, pady=10)

# Mostramos la ventana
ventana.mainloop()