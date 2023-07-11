# WGAN_GP-Labeled-acceleration-data-augmentation-for-structural-damage-detection

The study is published in Journal of Civil Structural Health Monitoring: https://link.springer.com/article/10.1007/s13349-022-00627-8.

# Code
Used these scripts around 2 years ago, some functions might have been deprecated, let me know if any of them doesn't work by creating a ticket in the issues. This study was also the first times I was learning about GANs, so the scripts below were typed in very simple and beginner level.

DCNN_and_training.py: provides the training of DCNN model and its training procedure

metrics.py: provides some of the metrics used in the study

model_generator_critic.py: provides the generator and critic models

trainWGAN-GP.py: provides the training procedure of WGAN-GP (or WDCGAN-GP) since it includes deep convolutional networks

utils.py: provides the gradient penalty

# Dataset
The dataset used in the paper can be obtained here: http://onur-avci.com/benchmark/
I shared two folders in the repo, named a01 (undamaged) and a11 (damaged), where both a01 and a11 folders consists of 29 amount of vibration data sample csv file. Normally, the acceleration data samples are found in the form of 1024 second signals (each sample consist 262144 data sample) in the original dataset: http://onur-avci.com/benchmark/. But I divided each sample into 256 pieces, making each data sample consists of 1024 data points. You can do this division for the rest of the 227 amount of vibration data samples, which can be obtained in the original dataset.


# About the study
The objective of this study is to investigate the integrated use of 1-D GAN and 1-D DCNN to address the data scarcity problem for nonparametric vibration-based structural damage detection (level-I damage diagnostics). First, a GAN variant, 1-D Wasserstein Deep Convolutional Generative Adversarial Networks using Gradient Penalty (1-D WDCGAN-GP) is employed for synthetic labelled acceleration data generation to augment the training data with varying ratios. Subsequently, the 1-D DCNN model is trained with the synthetically augmented data and then tested on the unseen raw acceleration data to demonstrate the performance of the WDCGAN-GP. First figure below represents the objective of the study with a schematic diagram. It is a well-known fact that the imbalanced data classes are detrimental to the performance of DL models which lowers their prediction scores. This study aims to solve the imbalanced data classes problem due to the data scarcity issue in the SHM field. 

![Picture1](https://github.com/furknluleci/WGAN_GP-Labeled-acceleration-data-augmentation-for-structural-damage-detection/assets/63553991/5188fd6d-1a3c-4f6a-b4b1-e7f0e033e79c)

The figure below illustrates the methodology of the study. The amount of data samples in the damaged class is reduced with five different ratios (five different scenarios). Then, 1-D WDCGAN-GP (denoted as M1 in the figure) is used to augment the damaged class on five different levels. Subsequently, a 1-D DCNN model (denoted as M2 in the figure) is trained with a naturally balanced data set and tested on the unseen data set. Then, the same model is trained with the synthetically augmented data set for five different scenarios (different augmentation ratios/levels in each scenario) and tested on the same unseen data set. The damage detection with the 1-D DCNN model trained with the augmented data set gives a very similar prediction performance as the damage detection with a 1-D DCNN model trained with the naturally balanced data set.

The prediction results of 1-D DCNN are evaluated with regression and classification metrics. While the prediction scores for the synthetically augmented data set scenarios yielded 97% classification accuracy, the real data set yielded 100% classification accuracy. In other words, when the 1-D DCNN is trained with 1-D WDCGAN-GP augmented damaged class and real undamaged class, its prediction capability falls in 3% error margin compared to the benchmark case (Scenario 0, real undamaged and real damaged classes). Thus, from the perspective of the 1-D DCNN model, the augmented damaged class is almost indistinguishable from the real damaged data sets.


![Blank diagram](https://github.com/furknluleci/WGAN_GP-Labeled-acceleration-data-augmentation-for-structural-damage-detection/assets/63553991/f6fa2da7-0716-41ca-80bb-8a7ee4e8506d)

