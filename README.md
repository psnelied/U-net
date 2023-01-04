# U-net
This repository is a pipeline for 68 Facial Landmarks detection. 3D-Landmarks detection  is useful for various purposes such as face tracking, face recognition, and in our case as a precursor to action units detection.  

The model we train is a U-net with an efficient net backbone , it is inspired by the paper DeCAFA( https://arxiv.org/abs/1904.02549 ). The model is trained with various parameters on 300W_LP and the final performance is benchmarked on AFLW2000. The reference metric is the Normalised Mean Error.



![alt text](https://github.com/psnelied/U-net/blob/main/unet.png)
