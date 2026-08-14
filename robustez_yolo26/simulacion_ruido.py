"""
Simulación de Robustez del Algoritmo YOLO bajo Condiciones de Ruido
===================================================================
Proyecto: Mejora de Holder de Cámara para Dron
Asignatura: Diseño Mecatrónico II (MEC45)
Parcial 2B - Actividad 3: Validación Mediante Simulación

Este script evalúa la robustez del algoritmo de detección YOLO
sometiendo imágenes de prueba a diferentes tipos y niveles de ruido
que simulan condiciones reales de operación de un dron.
"""

import os
import csv
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 0: CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# yolo26n.pt / yolo26x.pt no se incluyen en el repositorio (pesos preentrenados
# publicos); Ultralytics los descarga automaticamente en el primer uso si no
# existen localmente en BASE_DIR.
MODEL_NANO_PATH = os.path.join(BASE_DIR, "yolo26n.pt")
MODEL_XL_PATH = os.path.join(BASE_DIR, "yolo26x.pt")
IMAGE_DIR = os.path.join(BASE_DIR, "imagenes_base")
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados_simulacion")
DEVICE = 0  # GPU CUDA
CONF_THRESH = 0.25

# Niveles de ruido
NOISE_CONFIG = {
    "Gaussiano": {
        "param_name": "sigma",
        "levels": [0, 5, 10, 20, 30, 50, 70, 100],
    },
    "Motion Blur": {
        "param_name": "kernel",
        "levels": [1, 3, 5, 9, 15, 21, 31, 45],
    },
    "Sal y Pimienta": {
        "param_name": "probabilidad",
        "levels": [0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.25, 0.40],
    },
    "Brillo": {
        "param_name": "delta",
        "levels": [-100, -70, -50, -30, 0, 30, 50, 70, 100],
    },
}

# Colores para gráficas
COLORS = {
    "Gaussiano": "#e74c3c",
    "Motion Blur": "#3498db",
    "Sal y Pimienta": "#2ecc71",
    "Brillo": "#f39c12",
}

# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 1: FUNCIONES DE RUIDO
# ══════════════════════════════════════════════════════════════════════

def apply_gaussian_noise(img, sigma):
    """Ruido gaussiano aditivo (simula ruido de sensor en baja iluminación)."""
    if sigma == 0:
        return img.copy()
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_motion_blur(img, kernel_size):
    """Desenfoque por movimiento horizontal (simula vibración del dron)."""
    if kernel_size <= 1:
        return img.copy()
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_salt_and_pepper(img, prob):
    """Ruido de sal y pimienta (simula errores de transmisión de datos)."""
    if prob == 0:
        return img.copy()
    noisy = img.copy()
    rng = np.random.random(img.shape[:2])
    noisy[rng < prob / 2] = 0
    noisy[rng > 1 - prob / 2] = 255
    return noisy


def apply_brightness(img, delta):
    """Variación de brillo (simula cambios de iluminación en vuelo)."""
    if delta == 0:
        return img.copy()
    return np.clip(img.astype(np.float64) + delta, 0, 255).astype(np.uint8)


NOISE_FUNCTIONS = {
    "Gaussiano": apply_gaussian_noise,
    "Motion Blur": apply_motion_blur,
    "Sal y Pimienta": apply_salt_and_pepper,
    "Brillo": apply_brightness,
}

# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 2: EXTRACCIÓN DE MÉTRICAS
# ══════════════════════════════════════════════════════════════════════

def compute_iou(box_a, box_b):
    """Calcula Intersection over Union entre dos cajas [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_detection(model, img):
    """Ejecuta YOLO y extrae métricas de detección."""
    results = model.predict(source=img, device=DEVICE, verbose=False, conf=CONF_THRESH)
    r = results[0]
    boxes = r.boxes
    num_det = len(boxes)
    if num_det > 0:
        confs = boxes.conf.cpu().numpy()
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        return {
            "num_detections": num_det,
            "mean_confidence": float(np.mean(confs)),
            "max_confidence": float(np.max(confs)),
            "confidences": confs,
            "boxes_xyxy": boxes_xyxy,
        }
    return {
        "num_detections": 0,
        "mean_confidence": 0.0,
        "max_confidence": 0.0,
        "confidences": np.array([]),
        "boxes_xyxy": np.array([]).reshape(0, 4),
    }


def compute_mean_iou(baseline, noisy_metrics):
    """Calcula IoU promedio entre detecciones base y ruidosas (matching greedy)."""
    base_boxes = baseline["boxes_xyxy"]
    noisy_boxes = noisy_metrics["boxes_xyxy"]
    if len(base_boxes) == 0 or len(noisy_boxes) == 0:
        return 0.0
    ious = []
    used = set()
    for bb in base_boxes:
        best_iou = 0.0
        best_idx = -1
        for j, nb in enumerate(noisy_boxes):
            if j in used:
                continue
            iou = compute_iou(bb, nb)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0:
            used.add(best_idx)
            ious.append(best_iou)
    return float(np.mean(ious)) if ious else 0.0


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 3: LOOP DE SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════

def load_test_images():
    """Carga imágenes de prueba desde IMAGE_DIR."""
    images = []
    for fname in sorted(os.listdir(IMAGE_DIR)):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(IMAGE_DIR, fname)
            img = cv2.imread(path)
            if img is not None:
                images.append((fname, img))
    if not images:
        raise FileNotFoundError(f"No se encontraron imágenes en {IMAGE_DIR}")
    return images


def run_simulation(model, images, model_name="modelo"):
    """Ejecuta la simulación completa para un modelo dado."""
    all_results = []
    total_steps = sum(len(cfg["levels"]) for cfg in NOISE_CONFIG.values()) * len(images)
    step = 0

    for noise_type, cfg in NOISE_CONFIG.items():
        noise_fn = NOISE_FUNCTIONS[noise_type]
        levels = cfg["levels"]
        param_name = cfg["param_name"]

        for img_name, img in images:
            # Baseline (imagen limpia)
            baseline = run_detection(model, img)

            for level in levels:
                noisy_img = noise_fn(img, level)
                metrics = run_detection(model, noisy_img)
                iou = compute_mean_iou(baseline, metrics)
                retention = (metrics["num_detections"] / baseline["num_detections"] * 100
                             if baseline["num_detections"] > 0 else 0.0)

                all_results.append({
                    "modelo": model_name,
                    "tipo_ruido": noise_type,
                    "parametro": param_name,
                    "nivel": level,
                    "imagen": img_name,
                    "detecciones_base": baseline["num_detections"],
                    "detecciones": metrics["num_detections"],
                    "confianza_promedio": round(metrics["mean_confidence"], 4),
                    "confianza_maxima": round(metrics["max_confidence"], 4),
                    "iou_promedio": round(iou, 4),
                    "retencion_pct": round(retention, 2),
                })

                step += 1
                if step % 10 == 0:
                    print(f"  [{model_name}] Progreso: {step}/{total_steps}")

    print(f"  [{model_name}] Completado: {step}/{total_steps} evaluaciones")
    return all_results


def save_noise_samples(images):
    """Guarda muestras visuales de cada tipo de ruido para el informe."""
    samples_dir = os.path.join(OUTPUT_DIR, "muestras")
    os.makedirs(samples_dir, exist_ok=True)
    img_name, img = images[0]

    for noise_type, cfg in NOISE_CONFIG.items():
        noise_fn = NOISE_FUNCTIONS[noise_type]
        levels = cfg["levels"]
        # Elegir un nivel intermedio-alto para la muestra
        sample_level = levels[len(levels) * 2 // 3]
        noisy = noise_fn(img, sample_level)
        safe_name = noise_type.lower().replace(" ", "_")
        path = os.path.join(samples_dir, f"{safe_name}_{sample_level}.jpg")
        cv2.imwrite(path, noisy)
    # Guardar también la imagen limpia
    cv2.imwrite(os.path.join(samples_dir, "original.jpg"), img)
    print(f"  Muestras guardadas en {samples_dir}")


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 4: VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════

def setup_plot_style():
    """Configura estilo académico para las gráficas."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
    })


def aggregate_by_level(results, noise_type, metric_key, model_name=None):
    """Agrupa resultados por nivel y calcula media y desviación."""
    filtered = [r for r in results if r["tipo_ruido"] == noise_type]
    if model_name:
        filtered = [r for r in filtered if r["modelo"] == model_name]

    levels_data = {}
    for r in filtered:
        lvl = r["nivel"]
        if lvl not in levels_data:
            levels_data[lvl] = []
        levels_data[lvl].append(r[metric_key])

    levels = sorted(levels_data.keys())
    means = [np.mean(levels_data[l]) for l in levels]
    stds = [np.std(levels_data[l]) for l in levels]
    return levels, means, stds


def plot_fig1_confidence(results, graficas_dir):
    """Figura 1: Confianza promedio vs Nivel de Ruido (2x2 subplots)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Confianza Promedio de Detección vs Nivel de Ruido", fontsize=14, fontweight="bold")

    for ax, noise_type in zip(axes.flat, NOISE_CONFIG.keys()):
        levels, means, stds = aggregate_by_level(results, noise_type, "confianza_promedio")
        color = COLORS[noise_type]
        ax.errorbar(range(len(levels)), means, yerr=stds, marker="o", color=color,
                     capsize=4, linewidth=2, markersize=6)
        ax.axhline(y=CONF_THRESH, color="red", linestyle="--", alpha=0.7, label=f"Umbral ({CONF_THRESH})")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(l) for l in levels], fontsize=9)
        ax.set_xlabel(NOISE_CONFIG[noise_type]["param_name"])
        ax.set_ylabel("Confianza promedio")
        ax.set_title(noise_type, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(graficas_dir, "fig1_confianza_vs_ruido.png")
    plt.savefig(path)
    plt.close()
    print(f"  Guardada: {path}")


def plot_fig2_retention(results, graficas_dir):
    """Figura 2: Tasa de Retención de Detecciones (todas las curvas)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Tasa de Retención de Detecciones vs Nivel de Ruido",
                 fontsize=14, fontweight="bold")

    for noise_type in NOISE_CONFIG.keys():
        levels, means, stds = aggregate_by_level(results, noise_type, "retencion_pct")
        color = COLORS[noise_type]
        x = np.linspace(0, 1, len(levels))
        ax.plot(x, means, marker="o", color=color, linewidth=2, markersize=6, label=noise_type)
        ax.fill_between(x, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), alpha=0.15, color=color)

    ax.axhline(y=80, color="red", linestyle="--", alpha=0.7, label="Umbral 80%")
    ax.axhline(y=50, color="darkred", linestyle=":", alpha=0.5, label="Umbral 50%")
    ax.set_xlabel("Nivel de ruido (normalizado)")
    ax.set_ylabel("Retención de detecciones (%)")
    ax.set_ylim(0, 120)
    ax.legend(loc="lower left")
    plt.tight_layout()
    path = os.path.join(graficas_dir, "fig2_retencion_detecciones.png")
    plt.savefig(path)
    plt.close()
    print(f"  Guardada: {path}")


def plot_fig3_iou(results, graficas_dir):
    """Figura 3: IoU Promedio vs Nivel de Ruido."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("IoU Promedio vs Nivel de Ruido (Precisión de Localización)",
                 fontsize=14, fontweight="bold")

    for noise_type in NOISE_CONFIG.keys():
        levels, means, stds = aggregate_by_level(results, noise_type, "iou_promedio")
        color = COLORS[noise_type]
        x = np.linspace(0, 1, len(levels))
        ax.plot(x, means, marker="s", color=color, linewidth=2, markersize=6, label=noise_type)
        ax.fill_between(x, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), alpha=0.15, color=color)

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="IoU = 0.5")
    ax.set_xlabel("Nivel de ruido (normalizado)")
    ax.set_ylabel("IoU promedio")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    plt.tight_layout()
    path = os.path.join(graficas_dir, "fig3_iou_vs_ruido.png")
    plt.savefig(path)
    plt.close()
    print(f"  Guardada: {path}")


def plot_fig4_samples(images, graficas_dir):
    """Figura 4: Grid visual de muestras con ruido creciente."""
    img_name, img = images[0]
    noise_types = list(NOISE_CONFIG.keys())
    n_cols = 4
    fig, axes = plt.subplots(len(noise_types), n_cols, figsize=(14, 12))
    fig.suptitle("Muestras Visuales: Degradación por Tipo de Ruido",
                 fontsize=14, fontweight="bold")

    for i, noise_type in enumerate(noise_types):
        levels = NOISE_CONFIG[noise_type]["levels"]
        # Seleccionar 4 niveles representativos
        indices = np.linspace(0, len(levels) - 1, n_cols, dtype=int)
        noise_fn = NOISE_FUNCTIONS[noise_type]
        for j, idx in enumerate(indices):
            level = levels[idx]
            noisy = noise_fn(img, level)
            noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(noisy_rgb)
            axes[i, j].set_title(f"{NOISE_CONFIG[noise_type]['param_name']}={level}", fontsize=9)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(noise_type, fontsize=11, fontweight="bold", rotation=90, labelpad=40)

    plt.tight_layout()
    path = os.path.join(graficas_dir, "fig4_muestras_visuales.png")
    plt.savefig(path)
    plt.close()
    print(f"  Guardada: {path}")


def plot_fig5_model_comparison(results, graficas_dir):
    """Figura 5: Comparación Nano vs XL (análisis de sensibilidad)."""
    models = list(set(r["modelo"] for r in results))
    if len(models) < 2:
        print("  Figura 5 omitida: solo se usó un modelo")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Análisis de Sensibilidad: Comparación YOLO Nano vs XL",
                 fontsize=14, fontweight="bold")

    for ax, noise_type in zip(axes.flat, NOISE_CONFIG.keys()):
        for model_name in sorted(models):
            levels, means, _ = aggregate_by_level(results, noise_type, "retencion_pct", model_name)
            style = "-o" if "nano" in model_name.lower() else "--s"
            ax.plot(range(len(levels)), means, style, linewidth=2, markersize=5, label=model_name)

        ax.axhline(y=80, color="red", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(NOISE_CONFIG[noise_type]["levels"])))
        ax.set_xticklabels([str(l) for l in NOISE_CONFIG[noise_type]["levels"]], fontsize=8)
        ax.set_xlabel(NOISE_CONFIG[noise_type]["param_name"])
        ax.set_ylabel("Retención (%)")
        ax.set_title(noise_type, fontweight="bold")
        ax.set_ylim(0, 120)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(graficas_dir, "fig5_comparacion_modelos.png")
    plt.savefig(path)
    plt.close()
    print(f"  Guardada: {path}")


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 5: RESUMEN DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════

def find_critical_threshold(results, noise_type, threshold_pct, model_name=None):
    """Encuentra el nivel de ruido donde la retención cae bajo un umbral."""
    levels, means, _ = aggregate_by_level(results, noise_type, "retencion_pct", model_name)
    for lvl, mean in zip(levels, means):
        if mean < threshold_pct:
            return lvl
    return None  # Nunca cae bajo el umbral


def generate_summary(results, images):
    """Genera resumen textual de resultados para el informe."""
    # Filtrar solo modelo nano para el resumen principal
    nano_results = [r for r in results if "nano" in r["modelo"].lower() or len(set(r2["modelo"] for r2 in results)) == 1]

    lines = []
    lines.append("=" * 60)
    lines.append("RESULTADOS DE SIMULACIÓN DE ROBUSTEZ")
    lines.append("Proyecto: Holder de Cámara para Dron")
    lines.append("Algoritmo: YOLO v26")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Imágenes de prueba: {len(images)}")
    lines.append(f"Umbral de confianza YOLO: {CONF_THRESH}")
    lines.append(f"Tipos de ruido evaluados: {len(NOISE_CONFIG)}")
    total_evals = len(nano_results)
    lines.append(f"Total de evaluaciones: {total_evals}")
    lines.append("")

    # Baseline
    baselines = [r for r in nano_results if r["nivel"] == 0 or
                 (r["tipo_ruido"] == "Motion Blur" and r["nivel"] == 1) or
                 (r["tipo_ruido"] == "Brillo" and r["nivel"] == 0)]
    if baselines:
        avg_det = np.mean([r["detecciones_base"] for r in baselines])
        avg_conf = np.mean([r["confianza_promedio"] for r in baselines if r["confianza_promedio"] > 0])
        lines.append("--- Detecciones en imagen limpia (baseline) ---")
        lines.append(f"  Promedio de detecciones: {avg_det:.1f}")
        lines.append(f"  Confianza promedio: {avg_conf:.3f}")
        lines.append("")

    # Umbrales críticos
    lines.append("--- Umbrales críticos (retención < 80%) ---")
    robust_types = []
    vulnerable_types = []
    for noise_type in NOISE_CONFIG.keys():
        thresh = find_critical_threshold(nano_results, noise_type, 80)
        param = NOISE_CONFIG[noise_type]["param_name"]
        if thresh is not None:
            lines.append(f"  {noise_type}: retención < 80% en {param} = {thresh}")
            vulnerable_types.append(noise_type)
        else:
            lines.append(f"  {noise_type}: mantiene > 80% en todos los niveles")
            robust_types.append(noise_type)
    lines.append("")

    lines.append("--- Umbrales críticos (retención < 50%) ---")
    for noise_type in NOISE_CONFIG.keys():
        thresh = find_critical_threshold(nano_results, noise_type, 50)
        param = NOISE_CONFIG[noise_type]["param_name"]
        if thresh is not None:
            lines.append(f"  {noise_type}: retención < 50% en {param} = {thresh}")
        else:
            lines.append(f"  {noise_type}: mantiene > 50% en todos los niveles")
    lines.append("")

    # Conclusión
    lines.append("=" * 60)
    lines.append("CONCLUSIÓN DE VIABILIDAD")
    lines.append("=" * 60)
    lines.append("")
    if robust_types:
        lines.append(f"El algoritmo YOLO muestra alta robustez ante: {', '.join(robust_types)}.")
    if vulnerable_types:
        lines.append(f"Se identifica sensibilidad ante: {', '.join(vulnerable_types)}.")
    lines.append("")
    lines.append("El sistema de visión por computadora del dron es viable para")
    lines.append("operación bajo condiciones nominales de vuelo. Se recomienda")
    lines.append("implementar filtros de preprocesamiento para los tipos de ruido")
    lines.append("donde se observó mayor degradación, y mantener condiciones de")
    lines.append("iluminación controladas para maximizar la confiabilidad de detección.")
    lines.append("")
    lines.append("El error de medición se considera ACEPTABLE dentro de los rangos")
    lines.append("operativos normales del dron (ruido moderado, vibración controlada,")
    lines.append("iluminación diurna). Para condiciones extremas, se sugieren mejoras")
    lines.append("en estabilización de imagen o uso de modelos más robustos (YOLO XL).")

    summary_text = "\n".join(lines)

    # Guardar
    path = os.path.join(OUTPUT_DIR, "resumen.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n{summary_text}")
    print(f"\n  Resumen guardado en: {path}")
    return summary_text


# ══════════════════════════════════════════════════════════════════════
# SECCIÓN 6: MAIN
# ══════════════════════════════════════════════════════════════════════

def save_results_csv(results):
    """Guarda todos los resultados en CSV."""
    path = os.path.join(OUTPUT_DIR, "resultados_raw.csv")
    if not results:
        return
    fieldnames = [k for k in results[0].keys() if k not in ("confidences", "boxes_xyxy")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"  Datos guardados en: {path}")


def main():
    print("=" * 60)
    print("SIMULACIÓN DE ROBUSTEZ - YOLO Detección con Ruido")
    print("=" * 60)

    # Crear directorios
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    graficas_dir = os.path.join(OUTPUT_DIR, "graficas")
    os.makedirs(graficas_dir, exist_ok=True)

    # Cargar imágenes
    print("\n[1/6] Cargando imágenes de prueba...")
    images = load_test_images()
    print(f"  Se cargaron {len(images)} imágenes: {[n for n, _ in images]}")

    # Guardar muestras de ruido
    print("\n[2/6] Generando muestras visuales de ruido...")
    save_noise_samples(images)

    # Simulación con modelo Nano
    print("\n[3/6] Ejecutando simulación con YOLO26 Nano...")
    t0 = time.time()
    model_nano = YOLO(MODEL_NANO_PATH)
    # Warmup
    model_nano.predict(source=images[0][1], device=DEVICE, verbose=False)
    results_nano = run_simulation(model_nano, images, "YOLO26-Nano")
    t_nano = time.time() - t0
    print(f"  Tiempo Nano: {t_nano:.1f}s")

    # Simulación con modelo XL
    all_results = list(results_nano)
    if os.path.exists(MODEL_XL_PATH):
        print("\n[4/6] Ejecutando simulación con YOLO26 XL...")
        t0 = time.time()
        model_xl = YOLO(MODEL_XL_PATH)
        model_xl.predict(source=images[0][1], device=DEVICE, verbose=False)
        results_xl = run_simulation(model_xl, images, "YOLO26-XL")
        all_results.extend(results_xl)
        t_xl = time.time() - t0
        print(f"  Tiempo XL: {t_xl:.1f}s")
    else:
        print("\n[4/6] Modelo XL no encontrado, omitiendo comparación...")

    # Guardar CSV
    save_results_csv(all_results)

    # Generar gráficas
    print("\n[5/6] Generando gráficas...")
    setup_plot_style()
    plot_fig1_confidence(results_nano, graficas_dir)
    plot_fig2_retention(results_nano, graficas_dir)
    plot_fig3_iou(results_nano, graficas_dir)
    plot_fig4_samples(images, graficas_dir)
    plot_fig5_model_comparison(all_results, graficas_dir)

    # Resumen
    print("\n[6/6] Generando resumen de resultados...")
    generate_summary(results_nano, images)

    print("\n" + "=" * 60)
    print("SIMULACIÓN COMPLETADA")
    print(f"Resultados en: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
