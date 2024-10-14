<h1> Introducción a la IA - Proyecto Sustituto </h1>

<p>Daniel Cano Hernández (1193155952)</p>
<p>Ingeniería de Sistemas</p>

<hr/>

<p>Enlace de la competición de Kaggle: https://www.kaggle.com/competitions/playground-series-s4e8/</p>

<p><b>Objetivo:</b> El objetivo de esta competición es predecir si una seta es comestible (e) o venenosa (p) en función de sus características físicas.</p>

<hr/>

<h2>Fase 1</h2>
<p>La carpeta fase-1 contiene el Notebook con el modelo utilizado para generar las predicciones.</p>

<h3>Pasos para ejecutar el modelo:</h3>

<ol>
  <li>Descargar el archivo kaggle.json (token) en la página de configuración de Kaggle (https://www.kaggle.com/settings), una vez se haya iniciado sesión.</li>
  <li>Aceptar las reglas de la competición siguiendo los pasos: ingresar al enlace de la competición >> dirigirse a la pestaña "Rules" >> hacer click en el botón "I Understand and Accept" en la parte inferior.</li>
  <li>Abrir la carpeta fase-1 en este repositorio y abrir el Notebook "Intro a la IA - Mushoroom_c_Notebook.ipynb"</li>
  <li>Ejecutar cada una de las celdas en orden. Cada celda explica brevemente qué es lo que hace.</li>
</ol>
<p><b>Nota:</b> algunas de las celdas de código pueden tardar algunos minutos en ejecutarse completamente.</p>

<hr/>

<h2>Fase 2</h2>
<p>
  <strong>⛔⛔ IMPORTANTE:</strong> Debido a que los archivos .CSV eran demasiado grandes, se comprimieron para subirlos a GitHub. Se debe descomprimir los archivos test.zip y data/train.zip para que el modelo pueda funcionar correctamente.
</p>

<h3>Pasos para ejecutar el modelo:</h3>
<p><i>Todos los comandos se ejecutan estando ubicados en la carpeta fase-2</i></p>

Construir la imagen de Docker
``` bash 
docker build -t mushroom_img .
```

Ejecutar el contenedor de Docker
``` bash 
docker run -it mushroom_img
```

Hacer predicciones utilizando el modelo entrenado
``` bash 
python predict.py --input_file test.csv --model_file mushroom_model.pkl --output_file mushroom_preds.csv
```

Si deseas entrenar nuevamente el modelo, utiliza el siguiente comando
``` bash 
python train.py --data_file data/train.csv --model_file mushroom_model.pkl --overwrite_model
```

Puedes ver las predicciones realizadas en la consola con el siguiente comando
``` bash 
cat mushroom_preds.csv
```
