from ultralytics import YOLO
import cv2
import numpy as np
import argparse

from pathlib import Path

ROOT = Path.cwd()

class YOLO_Object_Detection:
    """
    Класс для обнаружения объектов с использованием модели YOLO.

    Этот класс предоставляет методы для загрузки модели YOLO, обработки видеофайлов и отображения результатов обнаружения.
    Он может быть использован для создания приложений, которые автоматически обнаруживают объекты в видеопотоке.
   """

    def __init__(self, model_path='yolo11n.pt'):
        """
        Инициализирует класс с путём к файлу модели YOLO по умолчанию.
        :param model_path: путь к файлу модели YOLO
        """

        self.model = YOLO(model_path)
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
            (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
            (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
        ]

    def detect_objects(self, input_video_path, output_video_path, class_to_display):
        """
        Обнаруживает объекты в указанном видеофайле и сохраняет результаты в новом видеофайле.
        :param input_video_path: путь к исходному видеофайлу
        :param output_video_path: путь к выходному видеофайлу
        """

        input_video_path = ROOT / input_video_path
        output_video_path = ROOT / output_video_path
       
        capture = cv2.VideoCapture(input_video_path)

        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            results = self.model(frame)[0]
            
            classes_names = results.names
            classes = results.boxes.cls.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
            
            for class_id, box, conf in zip(classes, boxes, results.boxes.conf):
                if conf > 0.5 and ( class_id == class_to_display or class_to_display != 0):
                    class_name = classes_names[int(class_id)]
                    color = self.colors[int(class_id) % len(self.colors)]
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Confidence: {conf:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            writer.write(frame)

    def parse_opt(self):
      """
      Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

      Returns:
          argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

      """
      parser = argparse.ArgumentParser()

      parser.add_argument("--input_video_path", type=str, default="crowd.mp4")
      parser.add_argument("--output_video_path", type=str, default="output.mp4")
      parser.add_argument("--class_to_display", type=str, default = 0) 

      opt = parser.parse_args()

      return opt

if __name__ == '__main__':
    detector = YOLO_Object_Detection()

    detector.detect_objects(**vars(detector.parse_opt()))