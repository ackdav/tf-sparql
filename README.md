# Predicting SPARQL Query Performance with TensorFlow

## TensorFlow installation on Debian SLURM cluster without sudo
The official installation guide to install TensorFlow doesn't provide an option without having sudo access. You can either use pip, virtualenv or docker, which all require sudo access to install in the first place. The trick is to install pip on a user level and then install TensorFlow.

`export PATH=/home/user/$USERNAME/.local/bin/:$PATH`  
`wget https://bootstrap.pypa.io/get-pip.py`

(get-pip.py is simply an entire copy of pip)

`python get-pip.py --user`

`.local/bin/pip install --upgrade --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl`

This installs TensorFlow on a user level. Please check for a newer version when installing due to breaking changes in upgrades.
