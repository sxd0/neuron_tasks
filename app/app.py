from __future__ import annotations

import gradio as gr
from PIL import Image

from finetuning.onnx_infer import ONNXPredictor


predictor = ONNXPredictor()


def classify(image: Image.Image):
    label, scores = predictor.predict(image)
    return label, scores


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Загрузите изображение"),
    outputs=[
        gr.Textbox(label="Предсказанный класс"),
        gr.Label(label="Вероятности по классам"),
    ],
    title="Классификация предметов через ONNX",
    description="Модель обучена на трех классах: mug, headphones, keyboard.",
)


if __name__ == "__main__":
    demo.launch()
