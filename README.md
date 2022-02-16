# Adversarial_Defense_Near_Sensor

### Adversarial Patch Generation

#### Step 1: Prepare Data
Create datasets folder in ./adversarial_patch_generation folder, then copy data.zip from dropbox folder and unzip it into datasets folder. 
You can use --train_size to select a certain number of training image to genenerate adversarial patch and use --test_size to test it. (e.g. --train_size = 1000   --test_size = 50)

#### Step 2: Patch Generation

In order to generte adversarial patch,  run Attack.py file: 
```bash
python Attack.py --
``` 
Check the args hyperparameters for more detils. 

The generated patch during each epoch will saved in ./training_pictures/ folder. 
The test image with adversarial patch will be saved in ./perturbed_image/ folder.

The attack success rate will be shown in the command line. 

###  Adversarial Patch Detection

#### Step 1: Model Definition

The small detection model structure is defined in small_net, try to change the code to generate models with different size. It also include the code which is used to copy weight parameters from pre-trained large model. 

#### Step 2: Model Finetune (Option)

Use train.py to fine-tune the small model and save the model into ./checkpoint folder. 

#### Step 3: Attack Activation Generation

Use attack_analysis to generate the activation during attack or original image. The activation arrays are stored in ./attention folder. 

#### Step 4: Detection Analysis

Refer the paper and write the detection method in ./attention/untitled.ipynb file. 
