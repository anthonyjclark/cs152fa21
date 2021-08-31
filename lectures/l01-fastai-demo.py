# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# ssh with forwarding
# conda
# notebook with port
# jupytext

# %%
from fastai.vision.all import *

# %%
path = untar_data(URLs.MNIST)
path.ls()
# Explore the directory of the downloaded data
# tree -d
# find testing/ -name '*.png' | wc -l

# %%
dls = ImageDataLoaders.from_folder(path, train="training", valid="testing")
dls.show_batch()

# %%
dls.valid_ds

# %%
learn = cnn_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(1)
# What is random accuracy?
# What is with the number in brackets? (batch size)

# %%
help(cnn_learner)

# %%
# ?cnn_learner

# %%
doc(cnn_learner)

# %%
doc(ImageDataLoaders.from_folder)

# %%
image_files = get_image_files(path/"testing")
image_files

# %%
learn.predict(image_files[0])

# %%
learn.show_results()

# %%
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15, 10))

# %%
interp.plot_confusion_matrix(figsize=(10, 10))

# %%
learn.export("day1.pkl")
