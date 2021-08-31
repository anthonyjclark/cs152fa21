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
from fastai.vision.all import *

# %%
path = untar_data(URLs.MNIST)

# %%
path.ls()

# %%
dls = ImageDataLoaders.from_folder(path, train='training', valid='testing')

# %%
dls.show_batch()

# %%
dls.valid_ds[4367]

# %%
learn = cnn_learner(dls, resnet18, metrics=accuracy)

# %%
learn.fine_tune(1)

# %%
60000/937

# %%
# ?cnn_learner

# %%
help(cnn_learner)

# %%
doc(cnn_learner)

# %%
doc(ImageDataLoaders.from_folder)

# %%
learn.show_results()

# %%
interp = ClassificationInterpretation.from_learner(learn)

# %%
interp.plot_top_losses(9)

# %%
interp.plot_confusion_matrix()

# %%
learn.export('day1-inclass')

# %%
