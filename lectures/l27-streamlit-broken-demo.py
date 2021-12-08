"""
ssh -L 8907:localhost:8907 dgx01
streamlit run lectures/l27-streamlit-demo.py --server.port 8907 -- /raid/cs152/data/SpoonOrFork/resnet18-1.pkl
"""

from io import BytesIO
import requests
import streamlit as st
import json
import os

from fastai.vision.all import PILImage

import torch
from torchvision.models import alexnet
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
import torchvision.transforms.functional as F

os.environ["TORCH_HOME"] = "/raid/cs152/pytorch"

model = None

with open("/raid/cs152/pytorch/imagenet_class_index.json") as json_file:
    labels = json.load(json_file)


sem_classes = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

preprocess = Compose(
    [
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(img, task):
    st.image(img, caption="Your image", use_column_width=True)

    x = preprocess(img).unsqueeze(0)

    if task == "Classification":
        with torch.inference_mode():

            yhat = model(x)

            probability = yhat.max().item()
            class_index = yhat.argmax().item()
            class_label = labels[str(class_index)][1]

        f"""
        prediction = '{class_label}' with a probability of {probability:.3f}%
        """

    else:

        with torch.inference_mode():

            yhat = model(x)["out"]

            print(yhat.shape)

            # normalized_masks = torch.nn.functional.softmax(yhat, dim=1)

            # dog_and_boat_masks = [
            #     normalized_masks[sem_class_to_idx[cls]] for cls in ("dog", "boat")
            # ]

            # st.image(F.to_pil_image(dog_and_boat_masks[0]))


def load_model(task):
    global model
    if task == "Classification":
        model = alexnet(pretrained=True, progress=True)
        model.eval()
    else:
        model = fcn_resnet50(pretrained=True, progress=True)
        model = model.eval()


image_option = st.radio("", ["Upload Image", "Image URL"])
model_option = st.radio("", ["Classification", "Segmentation"])

if image_option == "Upload Image":
    uploaded_file = st.file_uploader("Please upload an image.")


    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)  # type: ignore
        load_model(model_option)
        predict(img, model_option)

else:
    url = st.text_input("Please input a url.")

    if url != "":
        try:
            response = requests.get(url)
            pil_img = PILImage.create(BytesIO(response.content))  # type: ignore
            load_model(model_option)
            predict(pil_img, model_option)

        except Exception as e:
            st.text(f"Problem reading image from: '{url}' ({e})")
