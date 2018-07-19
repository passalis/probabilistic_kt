# Probabilistic Knowledge Transfer for Deep Neural Networks

In this repository we provide an implementation of a generic Probablistic Knowledge Transfer (PKT) method, as described in [our paper](?), which is capable of transferring the knowledge from a large and complex neural network (or any other model) into a smaller and faster one, regardless their architectures. The method was implemented and tested using PyTorch v.0.4.0.

To reproduce the results reported in out paper:
1. Train and evaluate the baselines model ([exp0_baseline_models.py](https://github.com/passalis/probabilistic_kt/blob/master/exp_cifar/exp0_baseline_models.py))
2. Use PKT, distillation and hints to transfer the knowledge into a smaller neural network and evaluate them ([exp1_retrieval_transfer.py](https://github.com/passalis/probabilistic_kt/blob/master/exp_cifar/exp1_retrieval_transfer.py))
3. Print the evaluation results ([exp2_print_results.py](https://github.com/passalis/probabilistic_kt/blob/master/exp_cifar/exp2_print_results.py))

Note that a pretrained Resnet-18 teacher model is also provided, along with the trained students. So you can directly use/evaluate the trained models and/or print the evaluation results.


If you use this code in your work please cite the following paper:

<pre>
@InProceedings{pkt_eccv,
author = {Passalis, Nikolaos and Tefas, Anastasios},
title = {Learning Deep Representations with Probabilistic Knowledge Transfer},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
year = {2018}
}
</pre>

