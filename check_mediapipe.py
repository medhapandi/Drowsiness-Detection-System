import importlib, traceback

try:
    importlib.import_module('mediapipe')
    print('mediapipe OK')
except Exception:
    traceback.print_exc()
