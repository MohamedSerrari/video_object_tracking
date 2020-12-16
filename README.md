## Video Object Tracking
This repository is a part of a project about object tracking. 
More specificaly about grasping object in the wild.

<p align="center">
  <img src="images/grasping_rice.gif?style=center" alt="Grasping in the wild" width=600px/>
</p>

For the first part of this project, we trained the MobileNetV2 architecture on a database of crops of 


## Setting up the project

### 1 - Clone the repository
```
git clone https://github.com/MohamedSerrari/video_object_tracking.git
```

### 2 - Install the dependencies of the project. 

* We recommend to use anaconda to manage your python environments.

* If you have a gpu, install `tensorflow-gpu` instead of `tensorflow` in order to benefit from GPU acceleration.

```
pip install -r requirements.txt
```

### 3 - Download the zip containing the labeled training and testing data:
```
https://dept-info.labri.fr/~mansenca/GITW_light/DB.zip
```

### 4 - Unzip the zip file into the cloned repo

The structure of your folder must now look like the structure shown here. 

``` text
.
├── train.py
├── eval.py
├── requirements.txt
├── README.md
├── DB
│   ├── train
│   │   ├── Bowl
│   │   ├── CanOfCocaCola
│   │   ├── MilkBottle
│   │   ├── Rice
│   │   └── Sugar
│   └── test
│       ├── Bowl
│       ├── CanOfCocaCola
│       ├── MilkBottle
│       ├── Rice
│       └── Sugar
├── logs
├── utils
│   ├── utils.py
│   ├── convert_checkpoint.ipynb
│   ├── data_exploration.ipynb
│   └── make_gifs.ipynb
├── weights
│   └── mobilenet_model.h5
└── results
    ├── conf_mat.gif
    ├── epoch_accuracy.jpg
    └── epoch_loss.jpg
```

## Analyzing the dataset

### Visualizing samples

### Class Distribution in Train and Test

These is a class imbalance between the training and testing data 

<p align="center">
  <img src="results/class_partition.jpg?style=center" alt="class_distribution" width=400px/>
</p>

### Conclusions about data:

There are multiples issues with the data:

* Many images have motion blurr.

* The dataset suffers from inclass heterogenity. This is a direct effect of using data captured in the wild.

* Different lighting conditions.

* Class imbalance between train and test set.

## Training The Model

* In order to train the model run the command

```
python train.py
```

During training, the script will save checkpoints of the model and show the accuracy on the training and evaluation set via `TensorBoard`. To follow the training you can launch `TensorBoard` using the command:

```
tensorboard --logdir=logs
```

Then Open a browser `http://localhost:6006/`


| Epoch Accuracy | Epoch Loss |
|:-------------------------:|:-------------------------:|
| ![](results/epoch_accuracy.jpg) | ![](results/epoch_loss.jpg) | 


<p align="center">
  <img src="results/conf_mat.gif?style=center" alt="conf_matrix" width=400px/>
</p>


## Evaluating The Model

In order to evaluate the model run the command

```
python eval.py
```

<pre>
               precision    recall  f1-score   support

         Bowl       0.91      0.99      0.95       166

CanOfCocaCola       0.98      0.98      0.98       180

   MilkBottle       0.98      0.98      0.98       177

         Rice       1.00      0.92      0.96       251

        Sugar       0.98      1.00      0.99       226

     accuracy                           0.97      1000

    macro avg       0.97      0.97      0.97      1000

 weighted avg       0.97      0.97      0.97      1000
</pre>

## Resources

`https://dept-info.labri.fr/~mansenca/GITW_light`