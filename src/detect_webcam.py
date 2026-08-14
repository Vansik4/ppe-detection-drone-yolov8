"""
Deteccion en tiempo real de casco (hardhat) y chaleco (vest) usando la camara.

Uso:
    yolo_env\\Scripts\\python.exe detect_webcam.py
    yolo_env\\Scripts\\python.exe detect_webcam.py --source 0 --conf-hardhat 0.65 --conf-vest 0.45
    yolo_env\\Scripts\\python.exe detect_webcam.py --weights runs/detect/runs/hardhat_vest_v1/weights/best.pt
"""
import argparse
import time

from window_capture import WindowCapture  # debe importarse antes que torch/cv2 (ver comentario en el modulo)

import cv2
from ultralytics import YOLO

HARDHAT_CLASS = 0
VEST_CLASS = 1
COLOR_HARDHAT = (255, 180, 0)
COLOR_VEST = (0, 200, 255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/runs/hardhat_vest_v1/weights/best.pt",
                         help="ruta al modelo entrenado (best.pt)")
    parser.add_argument("--source", default="0",
                         help="indice de camara (0,1,...), ruta/URL de video, o 'window' para capturar una ventana")
    parser.add_argument("--window-title", default="EasyCast",
                         help="texto (parcial) del titulo de la ventana a capturar, usado solo con --source window")
    parser.add_argument("--conf-hardhat", type=float, default=0.65,
                         help="umbral de confianza solo para casco (subelo si confunde frente/cabeza con casco)")
    parser.add_argument("--conf-vest", type=float, default=0.45, help="umbral de confianza solo para chaleco")
    parser.add_argument("--imgsz", type=int, default=640, help="tamano de entrada")
    parser.add_argument("--device", default=0, help="0 para GPU, 'cpu' para CPU")
    args = parser.parse_args()

    model = YOLO(args.weights)

    if args.source == "window":
        cap = WindowCapture(args.window_title)
        print(f"Capturando ventana: '{cap.window_title}'")
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if source == 0 else 0)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    floor_conf = min(args.conf_hardhat, args.conf_vest)
    prev_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(
            frame, conf=floor_conf, imgsz=args.imgsz, device=args.device, verbose=False
        )[0]
        annotated = frame.copy()
        if res.boxes is not None:
            for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                x1, y1, x2, y2 = map(int, box)
                if int(cls) == HARDHAT_CLASS and conf >= args.conf_hardhat:
                    color, label = COLOR_HARDHAT, f"hardhat {conf:.2f}"
                elif int(cls) == VEST_CLASS and conf >= args.conf_vest:
                    color, label = COLOR_VEST, f"vest {conf:.2f}"
                else:
                    continue
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Deteccion EPP - casco / chaleco (q para salir)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
