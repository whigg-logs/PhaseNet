# Phase Net

## Introduction
> A DNN method used for onset picking. 

The code is used for **onsetpicking**, which contains **Four DNN models**. 

|Model name|Num of parameters|Inver time|Download|
|:-:|:-:|:-:|:-:|
|3-layer CNN|52195|0.0088|[LOGS](#)|
|7-layer CNN|229283|0.025|[LOGS](#)|
|CNN+BRNN|476195|1.0|[LOGS](#)|
|WaveNet|2715651|0.17|[LOGS](#)|

This repository contains:

1. [Training code](train.py) used for training DNN.
2. [Testing code](test.py) used for test the trained model.
3. [Plots](plot.py) used for plot.


## Table of Contents

- [Phase Net](#phase-net)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [Usage](#usage)
  - [Traning](#traning)
    - [Usage](#usage-1)
  - [Environment](#environment)
  - [Maintainers](#maintainers)
  - [Credit](#credit)

## Quick Start 
1. Download the DNN weights. 
2. Prepare the sac file as any length of time. 
3. Run valid. 


### Usage
```
valid.py [-h] [--model MODLE]
         [--input DATA_FILE] 
         [--output OUT_FILE]
positional arguments:
  --input        Sac file
  --output       Phase file path 
optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model/wavenet 
``` 

## Traning 
1. Prepare the training data in .npz file.  
2. Run the train.py file. 

### Usage 
In order to prepare data, we wrote a script. You can run data.py. 
```
data.py [-h] [--catlog PATH_TO_CATLOG] 
         [--outfile PATH_TO_DATA] 
         [--n_process NUM_PROCESS] 
         [--if_gpu]
positional arguments:
  --catlog        path to catlog
optional arguments:
  -h, --help         show this help message and exit
  --data PATH        path to output, default data/ 
  --n_process        num of process 
  --if_gpu           if use GPU to process the data, default False. 
``` 
After the data is processed. You can run train.py. 

```
train.py [-h] [--data PATH_TO_DATA] 
         [--data PATH_TO_DATA] 
         [--n_gpu 1]
         [--batch_size 32]
positional arguments:
  --data        path to training data
optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model/wavenet
  --n_gpu            num of gpu used for training. 
  --batch_size       default 32. 
``` 

## Environment

This project uses [Tensorflow](http://tensorflow.com).

```sh
$ pip install tensorflow==1.13.1
```

## Maintainers

[@Cangye](https://github.com/cangyeone).


## Credit 
NN used for onset picking. 

**Abstract**
We present....