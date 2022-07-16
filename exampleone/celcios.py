# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# iportando datos 
teperatuta_df = pd.read_csv("celsius_a_fahrenheit.csv")

#visualiacion

sns.scatterplot(teperatuta_df['Celsius'],teperatuta_df['Fahrenheit'])
#cargando datos
X_train= teperatuta_df['Celsius']
y_train= teperatuta_df['Fahrenheit']

#creando el modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')
#entrenando el modelo
epochs_hist = model.fit(X_train, y_train, epochs = 100)

#Evaluando el modelo entrenado
epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de Pérdida durante Entrenamiento del Modelo')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend('Training Loss')
plt.show()

#prediciones