#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import onnxruntime as ort
from dt_computer_vision.camera.types import Pixel

from duckietown_messages.actuators.differential_pwm import DifferentialPWM
from solution.config import MODEL_PATH, CONF_THRESHOLD, STOP_DISTANCE, FORWARD_PWM

class MLModel:
    def __init__(self):
        print("Initializing MLModel")
        self.ground_projector = None

        if not MODEL_PATH.exists():
            raise FileNotFoundError("ONNX model not found (did you download your trained model?):", MODEL_PATH)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=sess_opts,
#            providers=["CUDAExecutionProvider", "CPUExecutionProvider"], 
            providers=["CPUExecutionProvider"], 

        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

        self.net_h = inp.shape[2]
        self.net_w = inp.shape[3]


    def _run_detector(self, img_bgr):
        x = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: x})[0]  # shape [1,N,6]
        return out[0]


    def _should_stop(self, detections: np.ndarray): 
        
        stop = False

        for x1, y1, x2, y2, score, _ in detections:

            print(f"Detection: {x1}-{x2}, {y1}-{y2}, {score}")

            # TODO : nous ne voulons pas prendre en compte les détections dont le score de confiance est inférieur à CONF_THRESHOLD (une valeur à définir dans config.py).

            # TODO : nous voulons arrêter la détection si un canard se trouve à une distance inférieure à STOP_DISTANCE.
            # Pour calculer si le canard est trop proche, nous devons convertir les coordonnées de pixels en
            # coordonnées du monde. Pour ce faire, vous pouvez utiliser l'objet `self.ground_projector` qui a
            # chargé le calibration extrinsèque de la caméra.
            # Plus précisément, si vous souhaitez projeter un objet de type `pix = Pixel(x=u, y=v)` sur un point du plan du sol,
            # vous pouvez d'abord le convertir en vecteur (`vec = self.ground_projector.camera.pixel2vector(pix)`) et
            # ensuite, vous pouvez intersecter ce vecteur avec le plan du sol (`self.ground_projector.vector2ground(vec)`).
            # Ce sera le point sur le plan du sol correspondant au pixel d'entrée.

        return stop


    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} does not match ONNX! Expected {self.net_h}x{self.net_w}"
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img


    def set_ground_projector(self, gp):
        self.ground_projector = gp
        

    def get_wheel_velocities_from_image(self, img: np.ndarray):
        try:
            detections = self._run_detector(img)
        except Exception as e:
            print(f"ONNX inference error {e}")
            return [DifferentialPWM(left=0.0, right=0.0), None]
        if self._should_stop(detections):
            return [DifferentialPWM(left=0.0, right=0.0), detections]
        else:
            return [DifferentialPWM(left=FORWARD_PWM, right=FORWARD_PWM), detections]
