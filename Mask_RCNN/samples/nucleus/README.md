# Semantic Segmentation and Cell Detection

This sample implements both the Cell Detection and Semantic Segmentation Code.
The goal is to segment individual nuclei in microscopy images.
The `nucleus.py` file contains the main parts of the code and the variations for each modality or functionality is also described.


## Command line Usage
Train a new model starting from ImageNet weights using `train` dataset (which is `stage1_train` minus validation set)
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```

Train a new model starting from specific weights file using the full `stage1_train` dataset
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5
```

Resume training a model that you had trained earlier
```
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last
```

Generate submission file from `stage1_test` images
```
python3 nucleus.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=<last or /path/to/weights.h5>
```


## Code Usage for the use of trained model

Save the testing Data in the /path/to/dataset/stage1_test/ and /path/to/weights/ is the downloaded model path.

### [MBA Cell Detection Model]
```
python3 nucleus0.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=</path/to/weights.h5>
```

### [BFI Cell Detection Model]
```
python3 nucleus1.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=</path/to/weights.h5>
```

### [Dendrite Detection Model]
```
python3 nucleus_dendrites.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=</path/to/weights.h5>
```

### [Passing Axons Detection Model]
```
python3 nucleus_passAxons.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=</path/to/weights.h5>
```
