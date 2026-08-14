# Detección de EPP mediante drones y Deep Learning (YOLOv8)

Sistema de visión artificial que usa un dron DJI Mini 4K y modelos YOLOv8 para
detectar en tiempo real el uso de casco y chaleco reflectivo por parte de
trabajadores, con envío automático de alertas por correo ante incumplimientos.

Proyecto Capstone — Ingeniería Mecatrónica, Institución Universitaria de
Barranquilla. Este repositorio es el Anexo B (código fuente) de la tesis.

## Estructura del repositorio

```
src/                        Código de la aplicación
  ppe_monitor.py             Programa principal: detección + lógica de cumplimiento + alertas SMTP
  detect_webcam.py           Pruebas de detección sobre cámara web / video local
  window_capture.py          Captura de la ventana de iVCam como fuente de video
  .env.example                Plantilla de configuración de correo (copiar como .env)

entrenamiento/               Entrenamiento del modelo YOLOv8s para casco/chaleco
  data.yaml                   Configuración del dataset (Roboflow)
  train_hardhat_vest_v1.log   Log de consola de la corrida de control (93 épocas)
  runs/
    hardhat_vest_v1-2/        Corrida FINAL (interrumpida en época 20/150) — la que usa ppe_monitor.py
    hardhat_vest_v1/           Corrida de control, completada por EarlyStopping en época 93/150 (sobreajustada, no usada)

evaluacion/                  Evaluación real sobre el conjunto de prueba (215 img, no vistas en entrenamiento)
  resumen_comparacion.md      Por qué se eligió el modelo de 20 épocas y no el de 93
  hardhat_vest_v1-2_test_final/          Métricas y gráficas del modelo final
  hardhat_vest_v1_test_sobreajustado/    Métricas y gráficas de la corrida de control

robustez_yolo26/             Análisis de robustez ante ruido (Sección 4.1.2 de la tesis)
  simulacion_ruido.py         Script de la simulación (YOLO26-Nano vs. YOLO26-XL)
  imagenes_base/              Imágenes de referencia usadas como entrada
  resultados_simulacion/      Resultados crudos, resumen y gráficas generadas
```

## Modelo final vs. modelo descartado

El entrenamiento del modelo de EPP se interrumpió manualmente en la época 20
de 150 planificadas (`hardhat_vest_v1-2`), por restricción de tiempo del
equipo. Para verificar si esto afectaba la calidad del modelo se entrenó una
segunda corrida bajo la misma configuración, dejándola completarse de forma
natural hasta la época 93 (`hardhat_vest_v1`).

Al evaluar ambas sobre el conjunto de prueba independiente, la corrida
completa resultó **peor** (66,1 % precisión / 59,2 % recall / 60,1 % mAP@50)
que la interrumpida (86,7 % / 81,3 % / 82,4 %) — un caso claro de
sobreajuste. Por eso el modelo de 20 épocas es el que se usa en producción.
Detalle completo en [`evaluacion/resumen_comparacion.md`](evaluacion/resumen_comparacion.md).

## Requisitos

```bash
pip install -r requirements.txt
```

Se probó con Python 3.12, CUDA 12.4 y una GPU NVIDIA (RTX 4070 Laptop, 8 GB).
Puede ejecutarse en CPU cambiando el parámetro `device` en las llamadas a
Ultralytics, aunque el rendimiento en tiempo real no está garantizado.

## Dataset

El dataset de entrenamiento (casco/chaleco) no se incluye en este repositorio
por su tamaño. Descárgalo desde Roboflow y colócalo según `entrenamiento/data.yaml`:

https://universe.roboflow.com/luanvan-wgmqx/hardhat-umya7/dataset/1

## Configuración de alertas por correo

1. Copia `src/.env.example` como `src/.env`.
2. Genera un [App Password de Gmail](https://myaccount.google.com/apppasswords) y complétalo en `SMTP_PASSWORD`.
3. **Nunca subas `src/.env` a GitHub** (ya está excluido en `.gitignore`).

## Uso

```bash
cd src
python ppe_monitor.py --source 0            # cámara local
python ppe_monitor.py --source window        # ventana de iVCam (ver window_capture.py)
python ppe_monitor.py --no-email             # sin envío de alertas
```

## Pesos preentrenados de propósito general

`yolov8s.pt` (detección de personas, COCO) y `yolo26n.pt` / `yolo26x.pt`
(simulación de robustez) no se incluyen en el repositorio: son pesos públicos
oficiales de Ultralytics que se descargan automáticamente en el primer uso.
