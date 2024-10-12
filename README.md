# ECG-JEPA: Learning General Representation of 12-Lead Electrocardiogram With a Joint-Embedding Predictive Architecture

### Installation
```console
(base) user@server:~$ conda create -name ecg_jepa python=3.9
(base) user@server:~$ conda activate ecg_jepa
(ecg_jepa) user@server:~$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
(ecg_jepa) user@server:~$ git clone https://github.com/sehunfromdaegu/ECG_JEPA.git
(ecg_jepa) user@server:~$ cd ECG_JEPA
(ecg_jepa) user@server:~/ECG_JEPA$ pip install -r requirements.txt
```

## Pretraining
### Pretrain the Model Yourself
To pretrain the ECG-JEPA model, run one of the following commands in the terminal:

For random masking\
python pretrain_ECG_JEPA.py --mask_type rand --mask_scale 0.6 0.7 --batch_size 128 --lr 2.5e-5 --data_dir_shao PATH_TO_SHAOXING --data_dir_code15 PATH_TO_CODE15

For multi-block masking:\
python pretrain_ECG_JEPA.py --mask_type block --mask_scale 0.175 0.225 --batch_size 64 --lr 5.5e-5 --data_dir_shao PATH_TO_SHAOXING --data_dir_code15 PATH_TO_CODE15

- PATH_TO_SHAOXING should be the path to the directory: 'path_to_data/ecg-arrhythmia/1.0.0/WFDBRecords'.
- PATH_TO_CODE15 should be the directory containing files like exams_part0.hdf5, exams_part1.hdf5, ..., exams_part17.hdf5.

### Pretrained Checkpoints
Pretrained checkpoints are available at the following links:

- ECG-JEPA (random masking): https://drive.google.com/file/d/1mh-XL0XOvvhFbhvuZ9c2KnTHa9B4F3Wx/view?usp=drive_link
- ECG-JEPA (multi-block masking): https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view?usp=drive_link

Please download the pretrained checkpoints and save them in the ./weights/

## Downstream Datasets
You can find information about the datasets used in this project here:

1. PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
2. CPSC2018: https://physionet.org/content/challenge-2020/1.0.2/sources/

In terminal, you can download datasets with the following command:

wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
or
wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/#files-panel

## Downstream Tasks
### Linear Evaluation 
Run the following command for the linear evaluation on ptbxl multi-label task:\
cd downstream_tasks python linear_eval.py --model_name ejepa_random --dataset ptbxl --data_dir PATH_TO_PTBXL --task multilabel

For ECG-JEPA multi-block masking, run:\
cd downstream_tasks python linear_eval.py --model_name ejepa_multiblock --dataset ptbxl --data_dir PATH_TO_PTBXL --task multilabel

Log files for the above two runs exist in './downstream_tasks/output/linear_eval/'.

Note that 'PATH_TO_PTBXL' should be the directory which contains 
1. records100
2. records500
3. index.html
4. LICENSE.txt
5. RECORDS 

'PATH_TO_PTBXL' can be replaced by 'PATH_TO_CPSC2018', which should be the directory containing the subdirectories g1, ..., g7. 

### Fine-Tuning
Run the following command for fine-tuning on the ptbxl multi-class task:

cd downstream_tasks
python finetuning.py --model_name ejepa_random --dataset ptbxl --data_dir PATH_TO_PTBXL --task multiclass

For ECG-JEPA multi-block masking, run:
python finetuning.py --model_name ejepa_multiblock --dataset ptbxl --data_dir PATH_TO_PTBXL --task multiclass
