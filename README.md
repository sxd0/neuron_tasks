## Название

Image classification with transfer learning, ONNX and Gradio

## Описание

Проект посвящён классификации изображений трёх классов: headphones, keyboard, mug.
Для решения задачи были дообучены две предобученные модели из библиотеки timm: resnet18 и efficientnet_b0. После сравнения моделей лучшая была экспортирована в ONNX и использована в локальном Gradio-приложении.

## Данные

Набор данных собран самостоятельно.
Классы:
- headphones
- keyboard
- mug

Структура:

- train
- val
- test

Установка
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Обучение
```
python experiments/train.py --data-dir data/raw/my_dataset --epochs 8 --freeze-epochs 3
```

Sweep
```
python experiments/train.py --data-dir data/raw/my_dataset --run-sweep --epochs 6 --freeze-epochs 2
```

Экспорт ONNX
```
python experiments/train.py --data-dir data/raw/my_dataset --epochs 8 --freeze-epochs 3 --export-onnx --onnx-file-name best_model.onnx
```

Запуск приложения
```
PYTHONPATH=. python app/app.py
```

Результаты

Лучшая модель: efficientnet_b0
Основной результат: test_accuracy = 1.0