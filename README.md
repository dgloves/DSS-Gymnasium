# DSS_Gymnasium
Welcome to DSS_Gymnasium, a project hub guide designed for assisting users in developing Deep Reinforcement Learning (DRL) environments for electric power distribution systems research using Python, OpenDSS (DSS), Gymnasium (Gym), and Stable-Baselines3 (SB3).

## Motivation
This repository is built to provide an instructional framework for building customized Gymnasium learning environments in OpenDSS for traditional optimization and control problems which may be converted 
into Markov Decision Processes (MDPs), and can be solved using model-free RL & DRL algorithms. 
The purpose of this repository is to provide a simple resource for both new and experienced Python users involved in electric power distribution systems research to aid in the process of custom learning environment creation for any optimization-based task which can be solved by Deep Reinforcement Learning algorithms with OpenDSS.
We understand that in the domain(s) of power distribution systems optimization and control, there exist many different types of operational grid challenges (state estimation, emergency restoration, voltage regulation, DER dispatch, etc.), each requiring different mathematical formulations specific to the application.  This translates into unique customizations for problem conversion into learning-based formulations (objective, constraints, etc.) as well as environmental structure (network, device, etc.).  Therefore, instead of attempting to create a single RL-based learning environment suitable for all problems (this is highly impractical) as many have attempted, we feel it is more important to teach users how to construct their own environment(s) using open-source tools based on his/her own specific grid optimization task.  This tutorial is built on foundational research in power systems at Washington State University in the SCALE Lab (Sustainable Climate-resilient Analytics for Large-scale Energy Systems) research group.  

## Toolkit Construction
The foundational framework utilizes OpenDSS, an open-source electric power distribution system simulator distributed by the Electric Power Research Institute [EPRI](https://sourceforge.net/p/electricdss/),
with the [Python](https://www.python.org/) programming language by way of [OpenDSSDirect](https://dss-extensions.org/OpenDSSDirect.py/#) and [DSS-Python](https://dss-extensions.org/DSS-Python/). OpenDSSDirect.py is a cross-platform Python package which
implements a "direct" library interface to a unique lower-level [implementation](https://github.com/dss-extensions/dss_capi) that allows users to automate OpenDSS processes using Pythonic functionality and other common packages.
The RL environment class structure is constructed through Farma's open-source package [Gymnasium](https://gymnasium.farama.org/), which is an updated fork based on the previous version from [OpenAi Gym](https://www.gymlibrary.dev/index.html), but with some improved customization capability.  For reinforcement learning implementation with neural networks, [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) provides RL algorithm selection via [PyTorch](https://pytorch.org/) and allows for user model flexibility and design customized to each individual user's DRL implementation.

![DSS-Gymnasium-Map](https://github.com/dgloves/DSS_Gymnasium/blob/main/dss_gymnasium_map.png?raw=True “DSS-Gymnasium-Framework”)

## Virtual Environment
Anaconda 3 distribution using Python ver. 3.10.13 (python 3.10+ applicable)

## Installation
Install the following open-source packages using pip or conda:
* Pytorch: 2.3.0 (CPU) - (GPU requires additional hardware and may be applicable)
* dss-python: 0.15.7
* gymnasium: 0.29.1
* matplotlib: 3.9.2
* numpy: 2.1.1
* opendssdirect-py: 0.9.4 
* pandas: 2.2.2
* scipy: 1.14.1
* stable-baselines 3: 2.3.2

## Contributing
Pull requests are welcome.  For significant changes, please open an issue first to discuss what you would like to change with respect to your particular branch.  This repo is also meant to be forked, allowing users to independently develop their own working environments. 
