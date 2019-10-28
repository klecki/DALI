#!/usr/bin/env python
# coding: utf-8

# # DALI expressions and arithmetic operators
#
# In this example, we will see how to use arithmetic operators in DALI Pipeline.

# In[1]:


import types
import collections
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.types import Constant

batch_size = 1

### Image examples
#
# Lets define a pipeline that will load some images

# In[6]:


from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.types import Constant
# import matplotlib.pyplot as plt

batch_size = 1
dogs = "images/dog"
cats = "images/kitten"

class BlendPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(BlendPipeline, self).__init__(batch_size, num_threads, device_id, seed=42)
        self.input_dogs = ops.FileReader(device="cpu", file_root="../images", file_list="dogs.txt")
        self.input_cats = ops.FileReader(device="cpu", file_root="../images", file_list="cats.txt")
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.resize = ops.Resize(resize_x=400, resize_y=400)
        self.uint8 = ops.Cast(dtype=types.DALIDataType.UINT8)
        self.int16 = ops.Cast(dtype=types.DALIDataType.INT16)

    def define_graph(self):
        dogs_buf, _ = self.input_dogs()
        cats_buf, _ = self.input_cats()
        images = self.decode([dogs_buf, cats_buf])
        dogs, cats = self.resize(images)
        result = dogs + 1
        dogs_i16 = self.int16(dogs)
        return dogs, cats, result, dogs_i16


# In[7]:


def display(output, cpu = True):
    data_idx = 0
    fig, axes = plt.subplots(1, len(output), figsize=(15, 15))
    if len(output) == 1:
        axes = [axes]
    for i, out in enumerate(output):
        img = out.at(data_idx) if cpu else out.as_cpu().at(data_idx)
        axes[i].imshow(img)


# In[13]:


pipe = BlendPipeline(batch_size=batch_size, num_threads=1, device_id=0)
pipe.build()


# In[14]:


output = pipe.run()
# display(output)

