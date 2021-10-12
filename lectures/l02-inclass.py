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
path = Path("/raid/cs152/SpoonOrFork")

# %%
path.ls()

# %%
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, item_tfms=Resize(224))

# %%
dls.show_batch()

# %%
len(dls.valid_ds), len(dls.train_ds)

# %%
learn = cnn_learner(dls, resnet18, metrics=accuracy)

# %%
# learn.lr_find()

# %%
learn.fine_tune(2, 0.0001)

# %%
resnet101()

# %%
