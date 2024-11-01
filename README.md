### Детекция и отрисовка людей. Компьютерное зрение
Для детекции людей в данном проекте используется модель YOLOv11 (You Only Look Once) с предобученными весами.

![image](https://github.com/user-attachments/assets/9960275b-b614-42b9-b39c-c4c161fcb355)

#### Установка
1. Скопируйте этот репозиторий:
   ```
   git clone https://github.com/IvanKarpovich/computer_vision.git
   cd computer_vision
   ```
2. Создайте виртуальную среду:
   ```
   conda create --name computer_vision
   conda activate computer_vision
   ```
   Установите в нее необходимые пакеты:
   ```
   pip install -r requirements.txt
   ```
4. Загрузите веса модели YOLOv1: yolo11n.pt
   (скрипт автоматически загружает модель YOLO, если она не найдена в директории проекта)

#### Использование
Запустите скрипт, используя следующую команду:
python main.py [options]

#### Опции командной строки
* `--classes`: идентификаторы классов для обнаружения (по умолчанию: обнаруживаются только люди).
  0 - только люди
  1 - все доступные классы
* `--input_video_path`: название видеофайла в директории с проектом (по умолчанию: «input.mp4»).
* `--output_video_path`: название видеофайла с результатами детекции (по умолчанию: «output.mp4»).

#### Пример использования
   Обнаружение только определённых классов (например, людей) в видеофайлах:
   ```
   python main.py --classes 0  --input_video_path crowd.mp4 --output_video_path detect.mp4
   ```

#### Структура проекта
* `main.py`: основной скрипт, содержащий все классы и функции.
* `YOLO_Object_Detection`: класс для работы с моделью YOLO, а также загрузки данных и параметров.

#### Формат вывода
Обработанные видео сохраняются в указанной выходной директории. 
В каждом кадре отображается обнаруженные объекты с ограничивающими рамками и метками.

#### Возможные улучшения
Можно в последствии добавить возможность не только детекции людей, но и сегментации.
