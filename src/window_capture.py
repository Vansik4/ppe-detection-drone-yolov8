"""
Captura el contenido de una ventana especifica de Windows (por titulo) como
frames BGR compatibles con OpenCV, en vez de usar una camara fisica.

Util para apuntar la deteccion a una app como EasyCast/OBS/DJI Fly que
esta reflejando la pantalla del celular o el video del dron en una ventana
de escritorio, sin necesidad de una camara virtual.

Uso directo:
    yolo_env\\Scripts\\python.exe window_capture.py --title EasyCast
"""
import argparse
import ctypes
import os
import sys

# PyTorch/OpenCV agregan sus propios directorios de busqueda de DLLs al importarse,
# lo que puede romper la carga de las DLLs de pywin32 si se importan despues.
# Registrar aqui explicitamente la carpeta de pywin32 evita ese conflicto sin
# importar cuidado el orden de imports en los scripts que usan este modulo.
_pywin32_dll_dir = os.path.join(sys.prefix, "Lib", "site-packages", "pywin32_system32")
if os.path.isdir(_pywin32_dll_dir):
    os.add_dll_directory(_pywin32_dll_dir)

import numpy as np
import win32con
import win32gui
import win32ui

ctypes.windll.user32.SetProcessDPIAware()


def find_window(title_substr):
    title_substr = title_substr.lower()
    matches = []

    def _enum(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            if title_substr in win32gui.GetWindowText(hwnd).lower():
                matches.append(hwnd)

    win32gui.EnumWindows(_enum, None)
    return matches[0] if matches else None


class WindowCapture:
    """Interfaz compatible con cv2.VideoCapture (isOpened/read/release/set)."""

    def __init__(self, title_substr):
        self.title_substr = title_substr
        self.hwnd = find_window(title_substr)
        if self.hwnd is None:
            raise RuntimeError(
                f"No se encontro ninguna ventana visible con titulo que contenga '{title_substr}'."
            )
        self.window_title = win32gui.GetWindowText(self.hwnd)

    def isOpened(self):
        return self.hwnd is not None and win32gui.IsWindow(self.hwnd)

    def set(self, *_args, **_kwargs):
        pass  # no aplica a captura de ventana; se mantiene por compatibilidad

    def read(self):
        if not self.isOpened():
            return False, None

        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        # GetClientRect da coordenadas relativas; convertir a pantalla para el tamano real
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            return False, None

        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        # PrintWindow (flag 3 = PW_RENDERFULLCONTENT) captura aun si esta tapada por otra ventana
        result = ctypes.windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 3)

        bmp_info = bitmap.GetInfo()
        bmp_bits = bitmap.GetBitmapBits(True)
        frame = np.frombuffer(bmp_bits, dtype=np.uint8).reshape(
            (bmp_info["bmHeight"], bmp_info["bmWidth"], 4)
        )
        frame = frame[:, :, :3]  # BGRA -> BGR

        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        if not result:
            return False, None
        return True, np.ascontiguousarray(frame)

    def release(self):
        pass


def main():
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="EasyCast", help="texto (parcial) del titulo de la ventana")
    args = parser.parse_args()

    cap = WindowCapture(args.title)
    print(f"Capturando ventana: '{cap.window_title}' (hwnd={cap.hwnd})")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer el frame (ventana minimizada o cerrada).")
            break
        cv2.imshow(f"Captura de ventana: {cap.window_title} (q para salir)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
