# Learning to Drive Anywhere via Regional Channel Attention (CoRL 2023 submission)

[//]: # (![]&#40;globaldrive.png&#41;)
<img src="globaldrive.png" alt="drawing" width="500"/>


## Updates
[08/10] Adding a minimal version for training and testing. 


## Dataset Setup
### nuScenes dataset
You can sign up and download the Full Dataset(v1.0) from the [nuScenes](https://www.nuscenes.org/nuscenes) official website. We follow nuScenes devkit github [repo](https://github.com/nutonomy/nuscenes-devkit) to build the dataset we use in ```datasets/realworld_data/nuscenes_dataset.py```

### Argoverse 2 dataset
You can download the Argoverse 2 Sensor Dataset from the [Argoverse 2](https://www.argoverse.org/av2.html#download-link) official website. We follow av2-api github [repo](https://github.com/argoverse/av2-api) to build the dataset we use in ```datasets/realworld_data/av2_dataset.py```.
This code includes a data preprocessing function which save the data as pickle file for faster later use.

### Waymo Open dataset
You can download the Perception Dataset from the [Waymo](https://waymo.com/open/download/) official website. We build the dataset we use in ```datasets/realworld_data/waymo.py```.
This code includes a data preprocessing function which read tf_records file for pytorch use.

### Real world driving dataset
We preprocess and merge these three dataset abovementioned into one dataset in ```datasets/realworld_data/driving_dataset.py```. For each city, we randomly select a small number of driving logs as validation set to monitor the training proces. We keep the validation set in a seperate folder and then set the test path as the root during training.



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
This repo is released under the MIT License (please refer to the LICENSE file for details).
