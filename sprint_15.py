#!/usr/bin/env python
# coding: utf-8

# ## Inicialización

# In[1]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns


# ## Carga los datos

# El conjunto de datos se almacena en la carpeta `/datasets/faces/`
# - La carpeta `final_files` con 7600 fotos
# - El archivo `labels.csv` con etiquetas, con dos columnas: `file_name` y `real_age`
# Dado que el número de archivos de imágenes es bastante elevado, se recomienda evitar leerlos todos a la vez, ya que esto consumiría muchos recursos computacionales. Te recomendamos crear un generador con ImageDataGenerator. Este método se explicó en el capítulo 3, lección 7 de este curso.
#
# El archivo de etiqueta se puede cargar como un archivo CSV habitual.

# In[2]:


df_labels = pd.read_csv('/datasets/faces/labels.csv')


datagen = ImageDataGenerator(rescale=1./255)

datagen_flow = datagen.flow_from_dataframe(
    dataframe=df_labels,
    directory='/datasets/faces/final_files/',
    x_col='file_name',
    y_col='real_age',
    target_size=(150, 150),
    batch_size=16,
    class_mode='raw',  # Regresion
    seed=12345
)


# ## EDA

# In[3]:


print(df_labels.info())
print()
print(df_labels.shape)
print()
print(df_labels['real_age'].describe())

# Histograma de las edades
plt.figure(figsize=(12, 6))
plt.hist(df_labels['real_age'], bins=30, edgecolor='black')
plt.title('Distribución de Edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Diagrama de caja para las edades
plt.figure(figsize=(12, 6))
sns.boxplot(x=df_labels['real_age'])
plt.title('Diagrama de Caja de Edades')
plt.xlabel('Edad')
plt.show()

print(df_labels.isnull().sum())


# In[4]:


# Mostrar 10 a 15 fotos para diferentes edades
features, target = next(datagen_flow)
plt.figure(figsize=(12, 12))
for i in range(10):
    plt.subplot(3, 5, i + 1)
    plt.imshow(features[i])
    plt.title(f"Edad: {int(target[i])}")
    plt.axis('off')
plt.show()


# ### Conclusiones

# In[ ]:


# ## Modelado

# Define las funciones necesarias para entrenar tu modelo en la plataforma GPU y crea un solo script que las contenga todas junto con la sección de inicialización.
#
# Para facilitar esta tarea, puedes definirlas en este notebook y ejecutar un código listo en la siguiente sección para componer automáticamente el script.
#
# Los revisores del proyecto también verificarán las definiciones a continuación, para que puedan comprender cómo construiste el modelo.

# In[5]:


# In[6]:


def load_train(path):
    """
    Carga la parte de entrenamiento del conjunto de datos desde la ruta.
    """

    # coloca tu código aquí
    df_labels = pd.read_csv(path + '/labels.csv')
    datagen = ImageDataGenerator(rescale=1./255)
    # Cargar las imágenes desde el directorio
    train_gen_flow = datagen.flow_from_dataframe(
        dataframe=df_labels,
        directory=path + '/final_files/',    # Ruta al directorio de entrenamiento
        x_col='file_name',
        y_col='real_age',
        target_size=(150, 150),       # Redimensionar imágenes a 150x150
        batch_size=16,                # Tamaño del lote
        class_mode='raw',           # Regresion
        seed=12345
    )

    return train_gen_flow


# In[7]:


def load_test(path):
    """
    Carga la parte de validación/prueba del conjunto de datos desde la ruta
    """

    # coloca tu código aquí
    df_labels = pd.read_csv(path + '/labels.csv')
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen_flow = datagen.flow_from_dataframe(
        dataframe=df_labels,
        directory=path + '/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(150, 150),
        batch_size=16,
        class_mode='raw',  # Regresión
        seed=12345
    )

    return test_gen_flow


# In[8]:


def create_model(input_shape):
    """
    Define el modelo
    """

    # coloca tu código aquí
    base_model = ResNet50(weights='imagenet',
                          include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(1)                   # Salida con una sola neurona para regresión
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='mse', metrics=['mae'])

    return model


# In[ ]:


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):
    """
    Entrena el modelo dados los parámetros
    """

    # # coloca tu código aquí
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        batch_size=batch_size
    )

    return model


path = '/datasets/faces'

# Carga los datos de entrenamiento y prueba usando las funciones definidas
train_data = load_train(path)
test_data = load_test(path)

# Crear el modelo
input_shape = (150, 150, 3)
model = create_model(input_shape)

# Entrenar el modelo
model = train_model(model, train_data, test_data, epochs=20)


# ## Prepara el script para ejecutarlo en la plataforma GPU

# Una vez que hayas definido las funciones necesarias, puedes redactar un script para la plataforma GPU, descargarlo a través del menú "File|Open..." (Archivo|Abrir) y cargarlo más tarde para ejecutarlo en la plataforma GPU.
#
# Nota: el script debe incluir también la sección de inicialización. A continuación se muestra un ejemplo.

# In[ ]:


# prepara un script para ejecutarlo en la plataforma GPU

init_str = """
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


import inspect

def load_train(path):
    """
Carga la parte de entrenamiento del conjunto de datos desde la ruta.
"""
    df_labels = pd.read_csv(path + '/labels.csv')
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen_flow = datagen.flow_from_dataframe(
        dataframe=df_labels,
        directory=path + '/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(150, 150),
        batch_size=16,
        class_mode='raw',  # Regresión
        seed=12345
    )
    return train_gen_flow

def load_test(path):
    """
Carga la parte de validación/prueba del conjunto de datos desde la ruta.
"""
    df_labels = pd.read_csv(path + '/labels.csv')
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen_flow = datagen.flow_from_dataframe(
        dataframe=df_labels,
        directory=path + '/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(150, 150),
        batch_size=16,
        class_mode='raw',  # Regresión
        seed=12345
    )
    return test_gen_flow

def create_model(input_shape):
    """
Define el modelo
"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  
        Dropout(0.5),             
        Dense(128, activation='relu'),
        Dense(1)                   # Salida con una sola neurona para regresión
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):
    """
Entrena el modelo dados los parámetros
"""
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        batch_size=batch_size
    ) 
    return model
"""

# Guardar el script

with open('run_model_on_gpu.py', 'w') as f:
    f.write(init_str)
    f.write('\n\n')

    # Agregar el código fuente de las funciones
    for fn_name in [load_train, load_test, create_model, train_model]:
        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')


# ### El resultado

# Coloca el resultado de la plataforma GPU como una celda Markdown aquí.

# In[ ]:


# ## Conclusiones

# In[ ]:


# # Lista de control

# - [ ]  El Notebook estaba abierto
# - [ ]  El código no tiene errores
# - [ ]  Las celdas con el código han sido colocadas en el orden de ejecución
# - [ ]  Se realizó el análisis exploratorio de datos
# - [ ]  Los resultados del análisis exploratorio de datos se presentan en el notebook final
# - [ ]  El valor EAM del modelo no es superior a 8
# - [ ]  El código de entrenamiento del modelo se copió en el notebook final
# - [ ]  El resultado de entrenamiento del modelo se copió en el notebook final
# - [ ] Los hallazgos se proporcionaron con base en los resultados del entrenamiento del modelo

# In[ ]:
