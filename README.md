### Random erasing layer for data augmentation

An implementation of random erasing data augmentation using Tensorflow and `tf.keras.layers.Layer`. The 'standard'
random erasing layer can be imported via
````python
from erasing.layer import ErasingLayer
````
Other similar layers which want to erase (one or multiple) rectangles from the image can be created by inheriting
from `ErasingBase` in `erasing.base_layer`. Run unit tests with pytest.


Loosely following the implementation here https://github.com/zhunzhong07/Random-Erasing/tree/master .

Based on Zhong et al. (2020). Random Erasing Data Augmentation. *Proceedings of the AAAI Conference on 
Artificial Intelligence (AAAI)* [(arXiv)](https://arxiv.org/abs/1708.04896)
