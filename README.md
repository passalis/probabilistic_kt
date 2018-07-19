# Probabilistic Knowledge Transfer for Deep Neural Networks

In this repository we provide an implementation of a Probablistic Knowledge Transfer (PKT) method, as described in ?, which is capable of transferring the knowledge from a large and complex neural network (or any other model) into a smaller and faster one. The method was implemented and tested using PyTorch v.0.4.0.

To reproduce the results reported in out paper:
1. Train and evaluate the baselines model (?)
2. Use PKT, distillation and hints to transfer the knowledge into a smaller neural network and evaluate them (?)
3. Print the evaluation results ()

Note that a pretrained Resnet-18 teacher model is also provided, along with the trained students. So you can directly use/evaluate the trained models and/or print the evaluation results.


If you use this code in your work please cite the following paper:

<pre>
@InProceedings{okt_eccv,
author = {Passalis, Nikolaos and Tefas, Anastasios},
title = {Learning Deep Representations with Probabilistic Knowledge Transfer},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
year = {2018}
}
</pre>

