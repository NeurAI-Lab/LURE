# Learn, Unlearn and Relearn: An Online Learning Paradigm for Deep Neural Networks

This repository contains the official implementation of the TMLR paper **Learn, Unlearn and Relearn: An Online Learning Paradigm for Deep Neural Networks** [[Paper](https://openreview.net/forum?id=WN1O2MJDST)] by **Vijaya Raghavan T Ramkumar, Elahe Arani and Bahram Zonooz** in [PyTorch](https://pytorch.org/). 

## Abstract
Deep neural networks (DNNs) are often trained on the premise that the complete training
data set is provided ahead of time. However, in real-world scenarios, data often arrive
in chunks over time. This leads to important considerations about the optimal strategy
for training DNNs, such as whether to fine-tune them with each chunk of incoming data
(warm-start) or to retrain them from scratch with the entire corpus of data whenever a new
chunk is available. While employing the latter for training can be resource intensive, recent
work has pointed out the lack of generalization in warm-start models. Therefore, to strike
a balance between efficiency and generalization, we introduce Learn, Unlearn, and Relearn
(LURE) an online learning paradigm for DNNs. LURE interchanges between the unlearning
phase, which selectively forgets the undesirable information in the model through weight
reinitialization in a data-dependent manner, and the relearning phase, which emphasizes
learning on generalizable features. We show that our training paradigm provides consistent
performance gains across datasets in both classification and few-shot settings. We further
show that it leads to more robust and well-calibrated models.

![alt text](https://github.com/NeurAI-Lab/LURE/blob/main/method_LURE.png) 

For more details, please see the [Paper](https://openreview.net/forum?id=WN1O2MJDST) and [Presentation](https://www.youtube.com/@neurai4080).

## Requirements

The code has been implemented and tested with `Python 3.8` and `PyTorch 1.12.1`.  To install the required packages: 
```bash
$ pip install -r requirements.txt
```


### Training 

Run [`LURE_main.py`](./LURE_main.py) for training the model in Anytiem framework with selective forgetting on CIFAR10 and CIFAR100. Run `ALMA.py` for training the model without selective forgetting which is the warm-started model. 

```
$ python .\LURE_main.py --data <data_dir> --log-dir <log_dir> --run <name_of_the_experiment> --dataset cifar10 --arch resnet18 \
--seed 10 --epochs 50 --decreasing_lr 20,40 --batch_size 64 --weight_decay 1e-4 --meta_batch_size 6250 --meta_batch_number 8 --snip_size 0.20 \
--save_dir <save-dir> --sparsity_level 1 -wb --gamma 0.1 --use_snip
```
For training the model with R-ImageNet, 

```
$ python ./LURE_main.py --data <data_dir> --imagenet_path <imagenet data path> --run <name_of_the_experiment> --dataset restricted_imagenet --arch resnet50 \
--seed 10 --epochs 50 --decreasing_lr 20,40 --batch_size 128 --weight_decay 1e-4 --meta_batch_size 6250 --meta_batch_number 8 --snip_size 0.20 \
--save_dir <save-dir> --sparsity_level 1 -wb --gamma 0.1 --use_snip

```
**Note Use `-buffer_replay`, `-no_replay` for training the model with buffer and without buffer data respectively. If no args is given then its by default model is trained with full replay.**




## Reference & Citing this work

If you use this code in your research, please cite the original works [[Paper](https://openreview.net/forum?id=WN1O2MJDST)] :

```


```

