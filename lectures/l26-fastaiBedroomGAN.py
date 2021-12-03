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

# %% [markdown]
# - https://github.com/nashory/gans-awesome-applications
# - https://deepmind.com/blog/article/wavenet-generative-model-raw-audio

# %%
from fastai.vision.all import *
from fastai.vision.gan import *
import torch

# %%
# Setting this to device 2 because it is not in use
torch.cuda.set_device(2)

# %%
path = untar_data(URLs.LSUN_BEDROOMS)

# %%
size = 64
bs = 128

dblock = DataBlock(
    blocks=(TransformBlock, ImageBlock),
    get_x=generate_noise,
    get_items=get_image_files,
    splitter=IndexSplitter([]),
    item_tfms=Resize((size, size), method=ResizeMethod.Pad),
    batch_tfms=Normalize.from_stats(
        torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])
    ),
)

dls = dblock.dataloaders(path, path=path, bs=bs)

# %%
dls.show_batch()

# %%
generator = basic_generator(out_size=size, n_channels=3, n_extra_layers=1)
critic = basic_critic(in_size=size, n_channels=3, n_extra_layers=1)
learn = GANLearner.wgan(dls, generator, critic, opt_func=Adam)

# %%
learn.recorder.train_metrics = True
learn.recorder.valid_metrics = False
learn.fit(5, 2e-4, wd=0.)

# %%
learn.show_results(max_n=30, ds_idx=0)

# %%
learn.fit(25, 2e-4, wd=0.)

# %%
learn.show_results(max_n=30, ds_idx=0)

# %%
learn.export('bedroomgan.pkl')

# %%
