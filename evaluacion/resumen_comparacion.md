# Comparación de generalización: modelo interrumpido vs. modelo completo

Evaluación real ejecutada con `yolo val` (Ultralytics 8.4.113) sobre el conjunto de
prueba independiente del dataset (`test/`, 215 imágenes, 445 instancias: 227 de
casco, 218 de chaleco), no usado en entrenamiento ni validación.
Parámetros: `imgsz=640 conf=0.25 iou=0.50` (mismo protocolo que describe la
Sección 4.3.2 de la tesis).

| Corrida | Épocas | Precisión | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| `hardhat_vest_v1-2` (interrumpida, **modelo final**) | 20 de 150 | 86,7 % | 81,3 % | 82,4 % | 60,3 % |
| `hardhat_vest_v1` (EarlyStopping natural) | 93 de 150 | 66,1 % | 59,2 % | 60,1 % | 31,7 % |

## Conclusión

El modelo que completó su entrenamiento de forma natural (93 épocas) generaliza
mucho peor sobre datos no vistos que el modelo interrumpido en la época 20 —
una caída de más de 20 puntos porcentuales en las tres métricas principales,
característica de sobreajuste (overfitting). El modelo interrumpido es, por
tanto, el adoptado como modelo final del proyecto y el que carga
`src/ppe_monitor.py` por defecto.

Carpetas con la evidencia gráfica de cada evaluación (matriz de confusión,
curvas precisión-recall, predicciones sobre muestras del test set):

- `hardhat_vest_v1-2_test_final/` — modelo final
- `hardhat_vest_v1_test_sobreajustado/` — corrida de control usada para la comparación
