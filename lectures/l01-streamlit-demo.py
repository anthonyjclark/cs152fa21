#!/opt/mambaforge/envs/cs152/bin/python

"""
Run with: streamlit run lectures/l01-streamlit-demo.py --server.port 8920
"""

from io import BytesIO
import requests
import streamlit as st
from fastai.vision.all import *


def predict(img):
    st.image(img, caption="Your image", use_column_width=True)
    pred, _, probs = learn_inf.predict(img)

    f"""
    prediction = {pred}

    probabilities = {probs}
    """


path = untar_data(URLs.MNIST)
learn_inf = load_learner(path / "day1.pkl")

option = st.radio("", ["Upload Image", "Image URL"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Please upload an image.")

    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)  # type: ignore
        predict(img)

else:
    url = st.text_input("Please input a url.")

    if url != "":
        try:
            response = requests.get(url)
            pil_img = PILImage.create(BytesIO(response.content))  # type: ignore
            predict(pil_img)

        except:
            st.text("Problem reading image from", url)  # type: ignore
