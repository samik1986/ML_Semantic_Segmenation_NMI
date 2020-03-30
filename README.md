# Code Repository for ML based Semantic Segmentation of brain cells. 

**SETUP The Environment:**

```
$ sudo apt-get install python-pip python-dev python-virtualenv

$ sudo apt-get install virtualenv

$ virtualenv ~/venv

$ source ~/venv/bin/activate

```
**INSTALL dependencies:**

```
(venv)$ cat requirements.txt | xargs -n 1 pip install

(venv)$ cat requirements3.txt | xargs -n 1 pip3 install
```

## Process Detection ##

### Step 1 ###

** Folder Structure **

Images folder : /data/Train/images/[...]_img.tif
Masks folder : /data/Train/masks2m/[...]_img.png [0,255]

** Training ALBU **

(venv)#src/preprocessing$ tif2rgb #Images_Folder <if images are grayscale>
(venv)#src/preprocessing$ renamer.py #Image_Folder
(venv)#src/preprocessing$ renamer.py #Masks_Folder png
(venv)#src$ preprocessing/folds4gen.py #Image_Folder
  
(venv)#src$ python3 train_eval.py resnet34_512_02_02.json --training

** The hyperparameters of the training are in src/resnet34_512_02_02.json **

