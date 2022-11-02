# codebrim_challenge
This repository was made to attend to the CODEBRIM challenge hosted on "https://dacl.ai/". We highly encourage the readers to visit "https://github.com/phiyodr/building-inspection-toolkit" for more information on automatic infrastructure inspection. 

# Dataset
Codebrim is a multi-classes multi-labels dataset of common infrastructure defects. A balanced version of the dataset is available with additional augmented images in the train dataset. 


Name      | Type        | Classes | train/val/test split
----------|-------------|---------------|-------------
CODEBRIM [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Mundt_Meta-Learning_Convolutional_Neural_Architectures_for_Multi-Target_Concrete_Defect_Classification_With_CVPR_2019_paper.html) [[Data]](https://zenodo.org/record/2620293#.YO8rj3UzZH4) | 6-class multi-target Clf  | |'Background', 'Crack', 'Spallation', 
'Efflorescence' , 'ExposedBars', 'CorrosionStain' |7,729 or 9,209 / 632 / 616 |



# Model
Our model is a Vision Transformer with 12 transformer encoder layers, 6 heads, an embedding dimension of 384 and a patchsize of 8. 
A class token is concatenated to the input patch sequence. The concatenated class tokens of the 4 last layers are used as features for the classification task.
The classifier is a simple linear layer. A sigmoid activation function converts the logits into probabilities and a 0.5 threshold converts the probabilities into predictions.
Our model was able to perform the following performance on the CODEBRIM test set:
```python
======Results======
Number of samples in test dataset:    632
Completely correct predicted samples: 490
ExactMatchRatio:                      77.53 %
F1-Score:                             0.90
Recall-NoDamage: 0.95
Recall-Crack: 0.92
Recall-Spalling: 0.91
Recall-Efflorescence: 0.85
Recall-BarsExposed: 0.91
Recall-Rust: 0.85
```

# Training
To deal with class imbalance, we apply multiple class-balancing tricks regarding the loss, the weight regularization constraints and the parameters freezing. 
We didn't use over-sampling by data augmentation to balance the training set.

# Usage
The jupyter notebook run.ipynb will run predictions and display images for any images in /data/.

To load the model
```python
model = build_model(pretrained_weights='./vit/weights/quantized_models.pth', img_size=224, num_cls=6, quantized=True)
```

To make predictions:
```python
labels_list =  ['NoDamage' , 'Crack', 'Spalling', 'Efflorescence', 'BarsExposed', 'Rust']
make_predictions(model, img_path, labels= labels_list)
```
![My Image](https://github.com/mpaques269546/codebrim_challenge/blob/main/datasets/data/image_0000761_crop_0000006.png)

```python
NoDamage       ........................................  0.00% 
Crack          ++......................................  5.16% 
Spalling       +++++++++++++++++++++++++++++++++++++++.100.00% 
Efflorescence  ........................................  0.02% 
BarsExposed    +++++++++++++++++++++++++++++++++++++++.100.00% 
Rust           +++++++++++++++++++++++++++++++++++++++. 98.58% 
inference time = 251.10 ms
```
  

