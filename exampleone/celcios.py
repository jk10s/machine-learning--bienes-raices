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
x_train= teperatuta_df['Celsius']
y_train= teperatuta_df['Fahrenheit']

#creando el modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='')