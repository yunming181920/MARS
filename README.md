## About

This is the official implementation of our NeurIPS paper:  
**[MARS: A Malignity-Aware Backdoor Defense in Federated Learning](https://arxiv.org/pdf/2509.20383)** 


## Installation
To run the code, firstly create a conda env with`python=3.8.19` and then install`torch=2.4.1` and `torchvision=0.15.2a0`.
* Install all dependencies using the requirements.txt in utils folder: `pip install -r utils/requirements.txt`.



## Experiments on CIFAR-10
```
python training.py --name cifar --params configs/cifar_fed.yaml
```
YAML files `configs/cifar_fed.yaml` stores the configuration for experiments.



