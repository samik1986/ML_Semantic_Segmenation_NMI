# Code Repository for ML based Semantic Segmentation of brain cells. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3928538.svg)](https://doi.org/10.5281/zenodo.3928537)

**SETUP The Environment:**

Install Kakadu

```
$ sudo apt-get install python-pip python-dev python-virtualenv

$ sudo apt-get install virtualenv

$ virtualenv ~/venv

$ source ~/venv/bin/activate

```
**INSTALL dependencies:**
Install opencv 3.4.5

```
(venv)$ cat requirements.txt | xargs -n 1 pip install

(venv)$ cat requirements3.txt | xargs -n 1 pip3 install

(venv)morse_code/src$ g++ ComputeGraphReconstruction.cpp -std=c++11 `pkg-config --cflags --libs opencv`

```

## Process Detection 

### Step 1 

**Folder Structure**

```
Images folder : /data/Train/images/[...]_img.tif

Masks folder : /data/Train/masks2m/[...]_img.png [0,255]
```

**Training ALBU**

```
(venv)src/preprocessing$ python3 tif2rgb #Images_Folder <if images are grayscale>
(venv)src/preprocessing$ python3 renamer.py #Image_Folder
(venv)src/preprocessing$ python3 renamer.py #Masks_Folder png
(venv)src$ python3 preprocessing/folds4gen.py #Image_Folder
  
(venv)src$ python3 train_eval.py resnet34_512_02_02.json --training
```

**The hyperparameters of the training are in src/resnet34_512_02_02.json**

### Step 2

**Generate the Data for training DM++**

```
(venv)morse_code$ python3 wrapperALBU.py 
(venv)morse_code$ python3 wrapperDM1.py <ensure the data is single channel 16-bit for MBA/ grayscale for BFI>
```

Input Folder Name (line 55), Output folder names (line 58) & trained model name (line 104) need to be updated in the 'wrapperALBU' code.
Input Folder Name (line 238) & Output folder names (line 246) need to be updated in the 'wrapperDM1' code.

**Training DM++**

```
(venv)DM_base$ python3 createData.py
(venv)DM_base$ python3 fullModel.py
```
Input folder names and .npy filename needs to be updated in the code.
Model name needs to be updated in the code.

**Testing DM++**

Generate ALBU and DM data for testing (same as training)

```
(venv)morse_code$ python3 wrapperALBU.py 
(venv)morse_code$ python3 wrapperDM1.py <ensure the data is single channel 16-bit for MBA/ grayscale for BFI>
```

Input Folder Name (line 55), Output folder names (line 58) & trained model name (line 104) need to be updated in the 'wrapperALBU' code.
Input Folder Name (line 238) & Output folder names (line 246) need to be updated in the 'wrapperDM1' code.

```
(venv)DM_base$ python3 tsting.py
```
Input Folder Name (line 56, 57), Output folder names (line 107) & trained model name (line 49) need to be updated in the 'wrapperALBU' code.
Create the output Directory.

**Evaluation**

```
(venv)ComputeScore$ python3 cal_F1.py #AnnotatedMaskFolder #PredictedOutputFolder .

```
Use the mask to eliminate the injection region (/Injection Removal) and calculate the score.


## Cell Detection

Details in Mask_RCNN folder.

## Semantic Categories Detection

Details in Mask_RCNN/samples/nucleus folder

## Bouton Detection

Details in bouton_code folder

