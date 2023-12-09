# Learning to Drive Anywhere  (CoRL 2023)
[Ruizhao Zhu](https://ruizhaoz.github.io/), Peng Huang, [Eshed Ohn-Bar](https://eshed1.github.io/) and [Venkatesh Saligrama](https://sites.bu.edu/data/). Boston University.

[//]: # (![]&#40;globaldrive.png&#41;)
<img src="globaldrive.png" alt="drawing" width="500"/>

This is official PyTorch/GPU implementation of the paper [Learning to Drive Anywhere](https://arxiv.org/abs/2309.12295):

```
@inproceedings{zhu2023learning,
  title={Learning to Drive Anywhere},
  author={Zhu, Ruizhao and Huang, Peng and Ohn-Bar, Eshed and Saligrama, Venkatesh},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```

## Updates
[10/08] Adding a minimal version for training and testing. More functions will coming soon!
### Catalog

- [x] Datasets preparation
- [x] Minimal training and testing code
- [ ] Detailed instructions and scripts for training with different settings (centralized, semi-supervised and federated).
- [ ] Carla data collection code
- [ ] Pretrained models.

## Getting Started
* To run CARLA and train the models, make sure you are using a machine with **at least** a mid-end GPU.
* We run our model on CARLA 0.9.13, install and environment needed [here](https://github.com/carla-simulator/carla/releases).
* Please follow requirement.txt to setup the environment.

## Dataset Preparation
* ### nuScenes dataset
You can sign up and download the Full Dataset(v1.0) from the [nuScenes](https://www.nuscenes.org/nuscenes) official website. We follow nuScenes devkit github [repo](https://github.com/nutonomy/nuscenes-devkit) to build the dataset we use in ```datasets/realworld_data/nuscenes_dataset.py```

* ### Argoverse 2 dataset
You can and download the Argoverse 2 Sensor Dataset from the [Argoverse 2](https://www.argoverse.org/av2.html#download-link) official website. We follow av2-api github [repo](https://github.com/argoverse/av2-api) to build the dataset we use in ```datasets/realworld_data/av2_dataset.py```.
This code includes a data preprocessing function which save the data as pickle file for faster later use.

* ### Waymo Open dataset
You can and download the Perception Dataset from the [Waymo](https://waymo.com/open/download/) official website. We build the dataset we use in ```datasets/realworld_data/waymo.py```.
This code includes a data preprocessing function which read tf_records file for pytorch use.

* ### Real world driving dataset
We preprocess and merge these three dataset abovementioned into one dataset in ```datasets/realworld_data/driving_dataset.py```



## Training
```bash
python train.py 
```
Other training settings is in different functions of ```driving_method.py```.

## Evaluation
```bash
python test.py
```

## License
This repo is released under the Apache 2.0 License (please refer to the LICENSE file for details).
