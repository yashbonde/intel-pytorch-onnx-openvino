#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import tempfile
import hashlib
import requests 
import numpy as np

from PIL import Image
import io

from openvino.inference_engine import IECore

import torchvision.transforms as transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def fetch(url):
  # efficient loading of URLS
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(
      url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp) and os.stat(fp).st_size > 0:
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    print("fetching", url)
    dat = requests.get(url).content
    with open(fp+".tmp", "wb") as f:
      f.write(dat)
    os.rename(fp+".tmp", fp)
  return dat


def get_image_net_labels():
  with open("labels.txt", 'r') as f:
    labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip()[10:] for x in f]
  return labels_map

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--model", help="Path to an .xml file with a trained model.", default = "fp32/resnet18.xml", type=str)
  parser.add_argument("--image", required=True, help="pass any image URL")
  args = parser.parse_args()
  print("-"*70)
  model_xml = args.model
  model_bin = os.path.splitext(model_xml)[0] + ".bin"

  labels = get_image_net_labels()

  # Plugin initialization for specified device and load extensions library if specified.
  print("Creating Inference Engine...")
  ie = IECore()

  # Read IR
  print("Loading network")
  net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")

  print("Loading IR to the plugin...")
  exec_net = ie.load_network(network=net, device_name="CPU", num_requests=2)
  print(f"exec_net: {exec_net}")
  print("-"*70)

  # define the image transformations to apply to each image
  image_transformations = transforms.Compose([
    transforms.Resize(256),                               # resize to a (256,256) image
    transforms.CenterCrop(224),                           # crop centre part of image to get (244, 244) grid
    transforms.ToTensor(),                                # convert to tensor
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),    # normalise image according to imagenet valuess
  ])

  image = Image.open(io.BytesIO(fetch(args.image)))  # load any image
  x = image_transformations(image) # [3, H, W]
  x = x.view(1, *x.size()).numpy()
  print(f"Input value: {x.shape}")

  # this is a bit tricky. So the input to the model is the input from ONNX graph
  # IECore makes a networkX graph of the "computation graph" and when we run .infer
  # it passes it through. If you are unsure of what to pass you can always check the
  # <model>.xml file. In case of pytorch models the value "input.1" is the usual
  # suspect. Happy Hunting!
  out = exec_net.infer(inputs={"input.1": x})

  # the output looks like this {"node_id": array()} so we simply load the output
  out = list(out.values())[0]
  print("Output Shape:", out.shape)
  print("-"*70)

  # keep the top 10 scores
  out = out[0]
  number_top = 10
  indices = np.argsort(out, -1)[::-1][:number_top]
  probs = out[indices]
  print(f"Top {number_top} results:")
  print("===============")
  for p, i in zip(probs, indices):
    print(p, "--", labels[i])
  print("-"*70)
