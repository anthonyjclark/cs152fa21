from io import BytesIO
import json
import requests

import streamlit as st

import torch
from fastai.vision.all import PILImage
from torchvision.models import alexnet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

"""
# NN Application Demo
Tuesday, December 7, 2021
"""

model = alexnet(pretrained=True, progress=True)

preprocess = Compose(
    [
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

with open("/raid/cs152/pytorch/imagenet_class_index.json") as json_file:
    labels = json.load(json_file)


def predict(img):
    img_processed = preprocess(img)

    model.eval()
    with torch.inference_mode():
        yhat = model(img_processed.unsqueeze(0)).squeeze()

    class_prob = yhat.max()
    class_index = yhat.argmax()
    class_label = labels[str(class_index.item())]

    st.image(img, caption="This is your image", use_column_width=True)

    f"""
    The predicted class label is **'{class_label[1].lower().replace('_', ' ')}'**

    That label as a probability of **{class_prob:.2f}%**
    """


image_option = st.radio("", ["Image Upload", "Image URL"])

if image_option == "Image Upload":

    uploaded_file = st.file_uploader("Please upload an image.")

    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        predict(img)
else:
    url = st.text_input("Please input a url.")

    if url != "":
        try:
            response = requests.get(url)
            pil_img = PILImage.create(BytesIO(response.content))
            predict(pil_img)

        except Exception as e:
            st.text(f"Problem reading image from: '{url}' ({e})")
