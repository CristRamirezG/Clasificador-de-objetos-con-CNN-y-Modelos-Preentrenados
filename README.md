### Descripción del Proyecto

Este proyecto aborda la creación de un modelo de clasificación de imágenes utilizando redes neuronales convolucionales (CNN) y modelos preentrenados. El objetivo es clasificar imágenes en categorías de objetos cotidianos como decoraciones, electrodomésticos, herramientas, ropa, entre otros. Este trabajo se desarrolló para la asignatura de **Inteligencia Artificial (2023)**.

Se generaron dos conjuntos de datos balanceados mediante diferentes técnicas: **submuestreo** y **balanceo dinámico durante el entrenamiento**. El análisis del modelo incluye métricas como precisión, recall, F1-score y una matriz de confusión para evaluar su rendimiento.

---

### Categorías del Dataset

El dataset se organizó en las siguientes categorías: 

- Decoración, electrodomésticos y muebles.
- Juguetes y juegos de mesa.
- Papelería.
- Cubiertos y utensilios de cocina.
- Electrodomésticos de entretenimiento.
- Jardinería.
- Baño.
- Ropa.
- Limpieza.
- Herramientas.

---

### Procesamiento de Imágenes

Se implementó una función para cargar y procesar imágenes con los siguientes pasos:

1. **Recorte y redimensionado:** Las imágenes se recortaron y ajustaron a dimensiones uniformes para garantizar consistencia.
2. **Normalización:** Los valores de píxeles se normalizaron al rango [0, 1].
3. **Filtrado de datos:** Se eliminaron imágenes duplicadas y aquellas con etiquetas inconsistentes.

---

### Estandarización de Etiquetas

Se utilizó `LabelBinarizer` de **scikit-learn** para convertir etiquetas multiclase en formato **one-hot encoding**. Esto facilitó la preparación de los datos para el modelo CNN.

---

### Creación y Entrenamiento del Modelo CNN

El modelo CNN se desarrolló con **TensorFlow/Keras**, incorporando:

- **Capas convolucionales y de pooling:** Para la extracción de características.
- **Dropout:** Para evitar sobreajuste.
- **Optimizador Adam:** Para la actualización eficiente de pesos.
- **EarlyStopping:** Para detener el entrenamiento automáticamente cuando la validación no mejora.

---


### Técnicas de Data Augmentation

Para mejorar el rendimiento, se aplicaron transformaciones como:

- **Redimensionado y recorte aleatorio:** Simulando diferentes perspectivas de los objetos.
- **Ajustes de brillo:** Para diversificar las condiciones de iluminación.
- **Pipeline optimizado con TensorFlow:** Utilizando técnicas como `AUTOTUNE` para maximizar la eficiencia.

---


## **Modelos Preentrenados:**

Se utilizó una variedad de modelos preentrenados, disponibles en bibliotecas como TensorFlow y PyTorch, para abordar el problema de clasificación de imágenes. Los modelos elegidos incluyen:

- **AlexNet**
- **MobileNetV2**
- **DenseNet169**
- **VGG19**
- **InceptionV3**
- **ResNet50**

Estos modelos fueron ajustados a los datos específicos mediante la modificación de sus capas de salida y el entrenamiento de las capas finales para adaptarse a las clases de interés.

---

## **Entrenamiento de Modelos Preentrenados**

### Consideraciones Previas

Para todos los modelos preentrenados, se asumió lo siguiente:
- Uso de un conjunto de datos aumentado en formato `tf.data.Dataset`.
- Configuración de pipelines eficientes para el almacenamiento en caché y preprocesamiento.
- Implementación de técnicas de `early stopping` para evitar sobreajuste.

### Fuente de Datos

Se seleccionó el conjunto de datos **ImageNet**, debido a su diversidad y relevancia para las clases a reconocer (como "Cubiertos", "Toallas", "Decoración", "Electrodomésticos" y "Juegos de mesa"). Este conjunto proporciona una base sólida para ajustar los modelos a las categorías objetivo.

---

## **Modelos Implementados**

### 1. MobileNetV2 (ImageNet)

```python
import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Cargar el modelo preentrenado
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Modificar la capa de salida
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(len(class_names), activation='softmax')(x)

model_MobileNetV2 = Model(inputs=base_model.input, outputs=x)

# Compilación y entrenamiento
optimizer = Adam(learning_rate=0.0001)
model_MobileNetV2.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history_MobileNetV2 = model_MobileNetV2.fit(train_ds, epochs=30, validation_data=test_ds, callbacks=[Early_Stopping], class_weight=pesos_clases_dict)
```

### 2. DenseNet169 (ImageNet)

```python
from tensorflow.keras.applications import DenseNet169

# Cargar el modelo preentrenado
base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Modificar la capa de salida
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(len(class_names), activation='softmax')(x)

model_DenseNet169 = Model(inputs=base_model.input, outputs=x)

# Compilación y entrenamiento
optimizer = Adam(learning_rate=0.0001)
model_DenseNet169.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history_DenseNet169 = model_DenseNet169.fit(train_ds, epochs=30, validation_data=test_ds, callbacks=[Early_Stopping], class_weight=pesos_clases_dict)
```

### 3. VGG19 (ImageNet)

```python
from tensorflow.keras.applications import VGG19

# Cargar el modelo preentrenado
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Modificar la capa de salida
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(len(class_names), activation='softmax')(x)

model_VGG19 = Model(inputs=base_model.input, outputs=x)

# Compilación y entrenamiento
optimizer = Adam(learning_rate=0.00001)
model_VGG19.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history_VGG19 = model_VGG19.fit(train_ds, epochs=100, validation_data=test_ds, callbacks=[Early_Stopping], class_weight=pesos_clases_dict)
```

### 4. InceptionV3 (ImageNet)

```python
from tensorflow.keras.applications import InceptionV3

# Cargar el modelo preentrenado
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Modificar la capa de salida
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(len(class_names), activation='softmax')(x)

model_InceptionV3 = Model(inputs=base_model.input, outputs=x)

# Compilación y entrenamiento
optimizer = Adam(learning_rate=0.0001)
model_InceptionV3.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history_InceptionV3 = model_InceptionV3.fit(train_ds, epochs=100, validation_data=test_ds, callbacks=[Early_Stopping], class_weight=pesos_clases_dict)
```

### 5. ResNet50 (ImageNet)

```python
from tensorflow.keras.applications import ResNet50

# Cargar el modelo preentrenado
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Modificar la capa de salida
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(len(class_names), activation='softmax')(x)

model_ResNet50 = Model(inputs=base_model.input, outputs=x)

# Compilación y entrenamiento
optimizer = Adam(learning_rate=0.0001)
model_ResNet50.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history_ResNet50 = model_ResNet50.fit(train_ds, epochs=30, validation_data=test_ds, callbacks=[Early_Stopping], class_weight=pesos_clases_dict)
```

---

## **Evaluación del Rendimiento**

Para cada modelo se evaluó el rendimiento utilizando:
- **Pérdida en el conjunto de prueba (`test_loss`)**
- **Precisión en el conjunto de prueba (`test_acc`)**
- Gráficas de pérdida y precisión durante el entrenamiento.
- Matrices de confusión y métricas de clasificación.

Estos resultados proporcionaron insights sobre el desempeño de cada modelo y su capacidad para generalizar en las clases objetivo.

Con los siguentes Resultados:

### Caso No balanceado
 los modelos preentrenados generaron en su mayoría mejores resultados que el modelo CNN propuesto. Los modelos DenseNet169 y ResNet50 destacaron con los mejores resultados de validación y F1-score. El modelo DenseNet169 tuvo un loss de validación de **0.3988** y un accuracy de validación de **0.8938**, con la mejor clasificación para la clase "ollas". Por otro lado, el modelo ResNet50 obtuvo un loss de validación de **0.3665** y un accuracy de validación de **0.9076**, siendo la mejor clasificación para la clase "muebles".

En relación a las recomendaciones para este caso, se sugiere mejorar la optimizacion o 'fine-tunning', de los modelos con el objetivo de obtener resultados superiores. Asimismo, es crucial mejorar la calidad del conjunto de datos original. Esto puede lograrse equilibrando la cantidad de datos, ya sea mediante programación o incorporando más imágenes al conjunto de datos.


![imagen](/image/1.png)
![imagen](/image/2.png)
![imagen](/image/3.png)



### Resultados con submuestreo

- ( - ) `percentage_diff_loss` indica mejora
- ( + ) `percentage_diff_loss` indica empeora
- ( + ) `percentage_diff_accuracy` indica mejora
- ( - ) `percentage_diff_accuracy` indica empeora

| Modelo pre-entrenado  | percentage_diff_loss | percentage_diff_accuracy |
|------------------------|----------------------|--------------------------|
| MobileNetV2           | 14.204472           | 1.887810                 |
| DenseNet169 (1)       | 2.081244            | -1.264265                |
| VGG19                 | 0.746269            | -3.924222                |
| InceptionV3           | 18.406220           | -3.971828                |
| ResNet50 (2)          | 14.815825           | -3.459674                |


En ese caso al igual que el modelo no balanceado los modelos pre entrenados con mejores resultados fueron DenseNet169 y ResNet50 pero presentan peores resultados de validación, por tanto se recomienda revisar la calidad o cantidad de las fotos, sobre todo en las clases que siguen presentando menor F1Score.

### Resultados con balanceo dinámico durante el entrenamiento

## Diferencia en puntos porcentuales de entrenamiento pesos balanceados y entrenamiento original.

( - ) percentage_diff_loss == mejora

( + ) percentage_diff_loss == empeora

( + ) percentage_diff_accuracy == mejora

( - ) percentage_diff_accuracy == empeora


| Modelo pre-entrenado      | percentage_diff_loss | percentage_diff_accuracy |
|---------------------------|----------------------|--------------------------|
| MobileNetV2               | -15.858860           | 10.868393                |
| DenseNet169 (1)           | 12.562688            | -2.069814                |
| VGG19                     | 20.187200            | -8.525034                |
| InceptionV3               | 17.240039            | -1.870454                |
| ResNet50 (2)              | 29.140518            | -3.052005                |


### result_df
En ese caso al igual que el modelo no balanceado, los modelos preentrenados con mejores resultados fueron DenseNet169 y ResNet50, pero presentan peores resultados de validación. Sin embargo, en este caso **se mejoró la predictibilidad** de las peores clases, ya que el F1Score presenta mejores resultados para estas mismas.

Por tanto, se recomienda revisar la calidad o cantidad de las fotos, sobre todo en las clases que siguen presentando menor F1Score.


### Conclusión general del caso.
El entrenamiento sin ningún tipo de balance en las clases presenta mejores resultados de validación, no así en la predictibilidad de sus clases (F1 score). Mejorando esto último en el entrenamiento con pesos balanceados, presentado peores resultados de validación, pero mejorando la capacidad de generalización del modelo, mejorando el F1 score de todas sus clases. Mientras que el entrenamiento con submuestreo empeora en los dos aspectos antes mencionados.
