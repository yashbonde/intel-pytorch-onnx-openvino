"""
Code to convert any model available in torchvision to ONNX runtime.
In practice any model can be replaced with `net` like:
```
from my_model_file import my_powerful_model as net
```
"""

import io
import os
import hashlib
import tempfile
import requests
import numpy as np
from PIL import Image
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def fetch(url):
  # efficient loading of URLS
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
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

if __name__ == "__main__":
  args = ArgumentParser()
  args.add_argument("--model", choices = ["resnet18", "resnet50"], default = "resnet18", type = str, help = "model to use")
  args.add_argument("--image", required=True, help = "pass any image URL")
  args = args.parse_args()
  print("-"*70)

  # import correct net
  """
  In practice you will load your model here with the parameters. It would look as follows:
  my_model = Model()
  my_model.load_state_dict(torch.load("path/to/model.pt"))
  """
  net = None
  if args.model == "resnet18":
    from torchvision.models.resnet import resnet18 as net
  elif args.model == "resnet50":
    from torchvision.models.resnet import resnet50 as net

  labels = get_image_net_labels()
  
  # load the model by downloading pretrained weights
  model = net(pretrained=True)
  model.eval()

  # define the image transformations to apply to each image
  image_transformations = transforms.Compose([
    transforms.Resize(256),                               # resize to a (256,256) image
    transforms.CenterCrop(224),                           # crop centre part of image to get (244, 244) grid
    transforms.ToTensor(),                                # convert to tensor
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),    # normalise image according to imagenet valuess
  ])

  image = Image.open(io.BytesIO(fetch(args.image)))  # load any image
  x = image_transformations(image) # [3, H, W]
  x = x.view(1, *x.size())
  print("Input value:", x.size())
  print("NOTE that in torch channel is first dim = 1")
  with torch.no_grad():
    out = model(x)
  print("Output shape:", out.size())

  # keep the top 10 scores
  number_top = 10
  probs, indices = torch.topk(out, number_top)
  print(f"Top {number_top} results:")
  print("===============")
  for p, i in zip(probs[0], indices[0]):
    print(p, "--", labels[i])

  # convert to ONNX
  print("-"*70)
  print("Now we are converting this pytorch model to ONNX model")
  torch.onnx._export(model, x, f'{args.model}.onnx', export_params=True)
  print(f"See in the folder we have a {args.model}.onnx file")
  print("-"*70)
