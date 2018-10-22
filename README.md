# Probabilistic Knowledge Transfer for Deep Neural Networks

In this repository we provide an implementation of a generic Probablistic Knowledge Transfer (PKT) method, as described in [our paper](https://arxiv.org/abs/1803.10837), which is capable of transferring the knowledge from a large and complex neural network (or any other model) into a smaller and faster one, regardless their architectures. The method was originally implemented using PyTorch v.0.2 and then tested with PyTorch v.0.4. If you face any issue training the models, please consider disabling CUDNN (torch.backends.cudnn.enabled = False).

To reproduce the results reported in our paper:
1. Train and evaluate the baselines model ([exp0_baseline_models.py](exp_cifar/exp0_baseline_models.py))
2. Use PKT, distillation and hints to transfer the knowledge into a smaller neural network and evaluate them ([exp1_retrieval_transfer.py](exp_cifar/exp1_retrieval_transfer.py))
3. Print the evaluation results ([exp2_print_results.py](exp_cifar/exp2_print_results.py))

Note that a pretrained Resnet-18 teacher model is also provided, along with the trained students. So you can directly use/evaluate the trained models and/or print the evaluation results.

To reproduce the results using the YouTube Faces dataset:
1. Donwload the [pre-processed dataset](https://www.dropbox.com/s/hlxmd1oofr8j0km/youtube_faces.tar.xz?dl=0) (or prepare it by yourself)
2. Use PKT to transfer the knowledge encoded in handcrafted features into a neural network and evaluate its performance ([exp_yf/exp1_transfer.py](exp_yf/exp1_transfer.py))
3. Print the evaluation results ([exp_yf/exp2_print_results.py](exp_yf/exp2_print_results.py))


If you use this code in your work please cite the following paper:

<pre>
@InProceedings{pkt_eccv,
author = {Passalis, Nikolaos and Tefas, Anastasios},
title = {Learning Deep Representations with Probabilistic Knowledge Transfer},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
year = {2018}
}
</pre>


