# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is created with reference to torchvision/utils.py.

Modify: torch.tensor -> jax.numpy.DeviceArray
If you want to know about this file in detail, please visit the original code:
    https://github.com/pytorch/vision/blob/master/torchvision/utils.py
"""
import math
import other
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from jax import random
import metrics

def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format_img=None):
  """Make a grid of images and Save it into an image file.

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
    fp:  A filename(string) or file object
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format_img(Optional):  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename,
      this parameter should always be used.
  """

  if not (
      isinstance(ndarray, jnp.ndarray)
      or (
          isinstance(ndarray, list)
          and all(isinstance(t, jnp.ndarray) for t in ndarray)
      )
  ):
    raise TypeError(f'array_like of tensors expected, got {type(ndarray)}')

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = (
      int(ndarray.shape[1] + padding),
      int(ndarray.shape[2] + padding),
  )
  num_channels = ndarray.shape[3]
  grid = jnp.full(
      (height * ymaps + padding, width * xmaps + padding, num_channels),
      pad_value,
  ).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = grid.at[
          y * height + padding : (y + 1) * height,
          x * width + padding : (x + 1) * width,
      ].set(ndarray[k])
      k = k + 1

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = np.array(jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8))
  im = Image.fromarray(ndarr.copy())
  im.save(fp, format=format_img)
  return im


def plot_images(test_ds, test_labs, test_obj,key =0, digits=range(10),
                 save_image=False, filename="digits.png", sub=False, v2_impute=True):

  # Define parameters for the 10x8 grid of images
  n_images = 8           # Number of images per row (for each digit)
  half_height = 14       # Half of the height of the image
  image_width = 28       # Width of the image
  n_rows = len(digits)            # Number of rows for the grid (one for each digit)

  # Select images for digits 0 to 9
  # Assume `test_labs` contains the labels and `test_ds` contains the images
  choice = random.randint(random.key(key), shape=(n_images,), minval=0, maxval=700)
  digit_indices = [np.where(test_labs == digit)[1][choice] for digit in digits]  # Get indices for each digit

  # Prepare to store images for each digit (top and bottom halves)
  top_half_list = []
  bottom_half_list = []
  in_top_labels =[]
  recon_logits = []

  for i in range(n_rows):
      indices = digit_indices[i]
      if v2_impute:
        in_top = test_ds[indices, 0:392]        # Top half (first 392 pixels)
        in_top = other.binarize_array(in_top, 0.02)  # Binarize the top half
        in_top_labels.append(other.binarize_array(test_ds[indices, 392:784] , 0.02))
        recon_bottom = test_obj[4]['recon_x2'][indices, :]  # Bottom half reconstructed
        recon_logits.append(recon_bottom)
        recon_bottom = other.binarize_array(1/(1 + jnp.exp(-recon_bottom)), 0.5)*255 # Sigmoid activation
        # recon_bottom = jnp.clip(recon_bottom, 0, 1)
        top_half_list.append(255*in_top)
        bottom_half_list.append(recon_bottom)
      else:
        in_top = test_ds[indices, 392:784]        # Top half (first 392 pixels)
        in_top = other.binarize_array(in_top, 0.02)  # Binarize the top half
        in_top_labels.append(other.binarize_array(test_ds[indices, 0:392] , 0.02))
        recon_bottom = test_obj[4]['recon_x1'][indices, :]  # Bottom half reconstructed
        recon_logits.append(recon_bottom)
        recon_bottom = other.binarize_array(1/(1 + jnp.exp(-recon_bottom)), 0.5)*255 # Sigmoid activation
        # recon_bottom = jnp.clip(recon_bottom, 0, 1)
        top_half_list.append(recon_bottom)
        bottom_half_list.append(255*in_top)

  # Stack all selected top and bottom halves
  in_top_all = np.vstack(top_half_list)          # Top halves for all digits
  recon_bottom_all = np.vstack(bottom_half_list) # Bottom halves for all digits
  labels = np.vstack(in_top_labels)
  logits = np.vstack(recon_logits)
  loss = metrics.bce_with_logits(logits, labels)

  # Reshape input data
  top_half_reshaped = in_top_all.reshape(-1, half_height, image_width)  # Reshape top halves
  bottom_half_reshaped = recon_bottom_all.reshape(-1, half_height, image_width)  # Reshape bottom halves

  # Convert grayscale images to RGB by stacking grayscale values into 3 channels
  top_half_rgb = jnp.stack([top_half_reshaped] * 3, axis=-1)
  bottom_half_rgb = jnp.stack([bottom_half_reshaped] * 3, axis=-1)

  # Create a red line (height 1 and width equal to image_width)
  red_line = np.full((1, image_width, 3), [150, 0, 0])   # Red line in RGB
  white_space = np.full((1, image_width, 3), [255, 255, 255])  # White space row

  # Prepare to create the 10x8 grid
  fig, axes = plt.subplots(n_rows, n_images, figsize=(16, 20))
  # fig, axes = plt.subplots(n_rows, n_images,constrained_layout = True)
  # fig, axes = plt.subplots(n_rows, n_images)


  # Fill the grid with the processed images
  for row in range(n_rows):
      for col in range(n_images):
          index = row * n_images + col  # Calculate the image index for the grid

          # Stack top half, red line, white space, and bottom half for the current image
          full_image_with_red_lines = jnp.vstack((
              top_half_rgb[index],    # Top half of the image
              red_line,               # Red line
              bottom_half_rgb[index]  # Bottom half of the image
          ))

          # Clip the pixel values to ensure they are within valid range
          full_image_with_red_lines = jnp.clip(full_image_with_red_lines, 0, 255)
          # Add the image to the corresponding subplot in the grid
          axes[row, col].imshow(full_image_with_red_lines)
          axes[row, col].axis('off')  # Hide the axes for a cleaner look

  # plt.subplot_tool()
  if sub:
    plt.tight_layout(rect=(0,0, 1, 0.7))
  # Show the 10x8 grid of images
  if save_image:
    plt.savefig(filename,bbox_inches='tight', pad_inches=0.02, transparent=False)
  # plt.subplots_adjust(hspace=0.1, wspace=0.1)
  plt.show()
  return labels, logits, loss