"""
Monitor de EPP (casco/chaleco) en tiempo real con alertas por correo.

Combina dos modelos:
  - Un YOLOv8 pre-entrenado en COCO para detectar personas.
  - El YOLOv8 entrenado en este proyecto para detectar casco/chaleco.

Por cada persona detectada revisa si hay un casco en la zona de la cabeza
y un chaleco en la zona del torso. Si a alguna persona le falta algo, se
guarda el frame anotado; cada ALERT_INTERVAL_SECONDS, si hubo al menos un
incumplimiento, se envia un correo con la ultima imagen y el detalle.

Uso:
    yolo_env\\Scripts\\python.exe ppe_monitor.py
    yolo_env\\Scripts\\python.exe ppe_monitor.py --source 0 --no-email
"""
import argparse
import os
import smtplib
import ssl
import threading
import time
from datetime import datetime
from email.message import EmailMessage

from window_capture import WindowCapture  # debe importarse antes que torch/cv2 (ver comentario en el modulo)

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

HARDHAT_CLASS = 0
VEST_CLASS = 1
COCO_PERSON_CLASS = 0

COLOR_OK = (0, 200, 0)
COLOR_VIOLATION = (0, 0, 255)
COLOR_HARDHAT = (255, 180, 0)
COLOR_VEST = (0, 200, 255)


def load_email_config():
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    cfg = {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "to": os.getenv("ALERT_TO", ""),
        "interval": int(os.getenv("ALERT_INTERVAL_SECONDS", "60")),
    }
    return cfg


def send_alert_email(cfg, image_bytes, summary_text):
    msg = EmailMessage()
    msg["Subject"] = "Alerta EPP: empleado sin equipo detectado"
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg.set_content(summary_text)
    msg.add_attachment(image_bytes, maintype="image", subtype="jpeg", filename="alerta_epp.jpg")

    context = ssl.create_default_context()
    with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
        server.starttls(context=context)
        server.login(cfg["user"], cfg["password"])
        server.send_message(msg)


def send_alert_email_async(cfg, image_bytes, summary_text):
    def _run():
        try:
            send_alert_email(cfg, image_bytes, summary_text)
            print(f"[{datetime.now():%H:%M:%S}] Correo de alerta enviado a {cfg['to']}")
        except Exception as exc:
            print(f"[{datetime.now():%H:%M:%S}] ERROR enviando correo: {exc}")

    threading.Thread(target=_run, daemon=True).start()


def box_center_in_region(person_box, item_box, region):
    px1, py1, px2, py2 = person_box
    height = py2 - py1
    if region == "head":
        ry1, ry2 = py1, py1 + height * 0.35
    else:  # torso
        ry1, ry2 = py1 + height * 0.15, py1 + height * 0.85

    ix1, iy1, ix2, iy2 = item_box
    cx, cy = (ix1 + ix2) / 2, (iy1 + iy2) / 2
    return px1 <= cx <= px2 and ry1 <= cy <= ry2


def check_compliance(person_boxes, hardhat_boxes, vest_boxes):
    """Devuelve lista de (person_box, faltantes) por cada persona detectada."""
    results = []
    for pbox in person_boxes:
        has_hardhat = any(box_center_in_region(pbox, h, "head") for h in hardhat_boxes)
        has_vest = any(box_center_in_region(pbox, v, "torso") for v in vest_boxes)
        missing = []
        if not has_hardhat:
            missing.append("casco")
        if not has_vest:
            missing.append("chaleco")
        results.append((pbox, missing))
    return results


def draw_annotations(frame, person_results, hardhat_boxes, vest_boxes):
    for box in hardhat_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HARDHAT, 1)
    for box in vest_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VEST, 1)

    any_violation = False
    for pbox, missing in person_results:
        x1, y1, x2, y2 = map(int, pbox)
        if missing:
            any_violation = True
            color = COLOR_VIOLATION
            label = "SIN " + " Y ".join(m.upper() for m in missing)
        else:
            color = COLOR_OK
            label = "OK"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return any_violation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppe-weights", default="../entrenamiento/runs/hardhat_vest_v1-2/weights/best.pt")
    parser.add_argument("--person-weights", default="yolov8s.pt")
    parser.add_argument("--source", default="0",
                         help="indice de camara (0,1,...), ruta/URL de video, o 'window' para capturar una ventana")
    parser.add_argument("--window-title", default="EasyCast",
                         help="texto (parcial) del titulo de la ventana a capturar, usado solo con --source window")
    parser.add_argument("--conf-hardhat", type=float, default=0.8,
                         help="umbral de confianza solo para casco (subelo si confunde frente/cabeza con casco)")
    parser.add_argument("--conf-vest", type=float, default=0.45,
                         help="umbral de confianza solo para chaleco")
    parser.add_argument("--conf-person", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    parser.add_argument("--interval", type=int, default=None,
                         help="segundos entre alertas (sobreescribe .env)")
    parser.add_argument("--no-email", action="store_true", help="no enviar correos (modo prueba)")
    args = parser.parse_args()

    cfg = load_email_config()
    if args.interval:
        cfg["interval"] = args.interval

    email_enabled = not args.no_email
    if email_enabled and not cfg["password"]:
        print("AVISO: SMTP_PASSWORD vacio en .env -> corriendo SIN envio de correo (modo prueba).")
        email_enabled = False

    person_model = YOLO(args.person_weights)
    ppe_model = YOLO(args.ppe_weights)

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

    window_start = time.time()
    pending_image_bytes = None
    pending_max_missing = set()
    pending_count = 0
    prev_t = time.time()

    print("Monitor EPP iniciado. Presiona 'q' para salir.")
    if not email_enabled:
        print("(Modo prueba: no se enviaran correos)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        person_res = person_model.predict(
            frame, classes=[COCO_PERSON_CLASS], conf=args.conf_person,
            imgsz=args.imgsz, device=args.device, verbose=False,
        )[0]
        person_boxes = person_res.boxes.xyxy.tolist() if person_res.boxes is not None else []

        floor_conf = min(args.conf_hardhat, args.conf_vest)
        ppe_res = ppe_model.predict(
            frame, conf=floor_conf, imgsz=args.imgsz, device=args.device, verbose=False,
        )[0]
        hardhat_boxes, vest_boxes = [], []
        if ppe_res.boxes is not None:
            for box, cls, conf in zip(
                ppe_res.boxes.xyxy.tolist(), ppe_res.boxes.cls.tolist(), ppe_res.boxes.conf.tolist()
            ):
                if int(cls) == HARDHAT_CLASS and conf >= args.conf_hardhat:
                    hardhat_boxes.append(box)
                elif int(cls) == VEST_CLASS and conf >= args.conf_vest:
                    vest_boxes.append(box)

        compliance = check_compliance(person_boxes, hardhat_boxes, vest_boxes)
        annotated = frame.copy()
        has_violation = draw_annotations(annotated, compliance, hardhat_boxes, vest_boxes)

        if has_violation:
            ok_enc, buf = cv2.imencode(".jpg", annotated)
            if ok_enc:
                pending_image_bytes = buf.tobytes()
            for _, missing in compliance:
                pending_max_missing.update(missing)
            pending_count += 1

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Monitor EPP - casco / chaleco (q para salir)", annotated)

        if now - window_start >= cfg["interval"]:
            if pending_image_bytes is not None:
                summary = (
                    f"Se detectaron incumplimientos de EPP en los ultimos "
                    f"{cfg['interval']} segundos.\n"
                    f"Frames con incumplimiento: {pending_count}\n"
                    f"Equipo faltante detectado: {', '.join(sorted(pending_max_missing)) or 'N/D'}\n"
                    f"Hora: {datetime.now():%Y-%m-%d %H:%M:%S}"
                )
                print(f"[{datetime.now():%H:%M:%S}] Incumplimiento detectado -> {summary!r}")
                if email_enabled:
                    send_alert_email_async(cfg, pending_image_bytes, summary)
            pending_image_bytes = None
            pending_max_missing = set()
            pending_count = 0
            window_start = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
