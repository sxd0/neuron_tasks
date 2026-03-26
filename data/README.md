# Описание датасета

## Классы
- mug
- headphones
- keyboard

## Структура
```text
data/raw/my_dataset/
    train/
        mug/
        headphones/
        keyboard/
    val/
        mug/
        headphones/
        keyboard/
    test/
        mug/
        headphones/
        keyboard/
```

## Метод сбора
Изображения взяты с открытых источников:
- разные ракурсы;
- разное расстояние до объекта;
- разное освещение;
- разный фон.

## Размер
- 28 изображений на класс в train;
- 6 изображений на класс в val;
- 6 изображений на класс в test.

## Предобработка
В обучении и инференсе используются:
- Resize / CenterCrop до 224x224;
- преобразование в tensor;
- нормализация по mean/std ImageNet.

Для train дополнительно используются аугментации:
- RandomResizedCrop;
- RandomHorizontalFlip;
- RandomRotation;
- ColorJitter.
