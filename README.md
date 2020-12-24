# intel-pytorch-onnx-openvino

Simple repo to demonstrate conversion of pytorch model to ONNX runtime and finally to the Openvino format. This process is super dependent on installing the `openvino` correctly. The instructions given below were valid and correct on the date of writing (24th December, 2020).

If you want an interactive version please use the [notebook](pytorch_to_openvino.ipynb).

## Installation

If you have python-3.9 installed you will have to install python-3.7.9 for this project to work. Follow the instructions below when building for first time (verified build on MacOS):
```
brew install pyenv                             # for syncing multitple versions on the machine
pip3 install virtualenv                        # virtual-environment maker, can use any other package
pyenv install 3.7.9                            # install the specific version
pyenv local 3.7.9                              # set local (this folder) version to 3.7.9
export LOCAL_PY_VER_PATH=`pyenv which python3` # set path for convinience
echo $LOCAL_PY_VER_PATH                        # [opt.] to check the path
$LOCAL_PY_VER_PATH -m venv .                   # using the path above build a virtual environment in this folder
source bin/activate                            # activate the local env
pip3 install -r requirements.txt               # install run dependencies
```

When coming back to this project simply activate the virtualenv as and the rest will be ready for you:
```
source bin/activate
```

#### Virtual Environment in Jupyter

For the interactive notebook please go here. In case you do not know how to setup virtual env in jupyter do the following:
```
jupyter kernelspec list                         # get existing kernel list
python3 -m ipykernel install --name=<name>      # install a new kernel remember the name
>>> Installed kernelspec intel-resnet-tf-demonstrator in /usr/local/share/jupyter/kernels/intel-resnet-tf-demonstrator
```

Now create a new file called `kernel.json` and add the following in it where `LOCAL_PY_VER_PATH` comes from above:
```
{
 "argv": [
  "<LOCAL_PY_VER_PATH>/bin/python3",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "intel-resnet-tf-demonstrator",
 "language": "python3.7"
}
```

Start a jupyter notebook and select the kernel `<name>`.

## ONNX Model

To get the model in the ONNX format first run the file `convert_pt2onnx.py`. In practice you can convert any network to using the `torch.onnx._export(model, sample_input, "path/to/onnx", export_params=True)`.

## From ONNX to Openvino

For this you must first have openvino installed on your system. Download from [here](https://software.intel.com/en-us/openvino-toolkit). Now I have added most of the requirements in my `requirements.txt` file, however you should also install those for OpenVino. After that run the following commands to setup environment variables:
```
export OPENVINO_FOLDER="path/to/openvino_2021"
cd $OPENVINO_FOLDER/bin
source setupvars.sh
cd $OPENVINO_FOLDER/deployment_tools/model_optimizer
pip3 install install -r requirements.txt
pip3 install install -r requirements_onnx.txt
```

If everything works correctly you will see an output like this:
```
[setupvars.sh] OpenVINO environment initialized
```

Now come back to this repo, Openvino environment setup works correctly only if you are in the `openvino_2021/bin` folder. Now we run the script `mo_onnx.py`:
```
mo_onnx.py --help                              # to get meanings of arguments to be passed
mkdir full_precision half_precision            # full_precision is FP36 and other is FP16
mo_onnx.py --input_model resnet18.onnx \
  --scale_values=[58.395,57.120,57.375] \
  --mean_values=[123.675,116.28,103.53] \
  --reverse_input_channels \
  --disable_resnet_optimization \
  --finegrain_fusing=False\
  --finegrain_fusing=False\
  --data_type=FP32 \
  --output_dir=fp32
```

If everything works correctly you should see 3 files in `/fp32` folder:
```
resnet18.bin
resnet18.mapping
resnet18.xml
```

And we do the same for fp16 as:
```
mo_onnx.py --input_model resnet18.onnx --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --reverse_input_channels --disable_resnet_optimization --finegrain_fusing=Fals --finegrain_fusing=Fals --data_type=FP16 --output_dir=fp16
```

## OpenVino Runtime

We need to write a simple python script that uses `openvino.inference_engine`, see [here](./classification.py).
