# syntorch
 sr and mask training on pytorch
 
Ref models:
[https://drive.google.com/drive/folders/1uSCfpc1W3IppaoU-NM9bZtVnT3E94oZK?usp=sharing](https://drive.google.com/drive/folders/1uSCfpc1W3IppaoU-NM9bZtVnT3E94oZK?usp=sharing)

Dataset:
[https://drive.google.com/drive/folders/10UEK2L5AxIZUD3Cw1FKPLOXUi4uHIHoV?usp=sharing](https://drive.google.com/drive/folders/10UEK2L5AxIZUD3Cw1FKPLOXUi4uHIHoV?usp=sharing)

## Note for FusionSMSR

There are 2 versions of FusionSMSR: 

- v1 replaced the large module with original SMSR, continuously performs sparse convolution and always returns dense feature and feature map separately. They are merged only in the last 1x1 conv, after passed through all stages

- v2 which separated each stage (layer as original paper) by perform sparse convolution to split into dense feature and sparse feature, and merges them immediately. The output of each stage will be similar to normal convolution and we can merge them with which from small modules. The implementations of v2 might have some mistakes which are resolved.


## Commands:

### Testing
#### For test FusionNet_7_2s_1
python test_random_gradso.py --template FusionNet_7_2s_1 --checkpoint <model_name>

#### For test FusionSM_7_4s_v2
```bash
python test_smsr.py --core FusionSM_7_4s_v2 --testset_dir <path/to/testset>
```

### Training

#### (SMSR) For training C branch with SR and S branch with KD
```bash
python train_SM_sr_S_kd --template FusionSM_7_4s_v2 --trainset_dir <path/to/trainset> --testset_dir <path/to/testset>
```

For training C branch with SR and S branch with KD loss:

python train_C_sr_S_kd.py --template <template_name>

ex: python train_C_sr_S_kd.py --template FusionNet_7_2s_1

for training C branch with SR and S branch with SR+KD loss:

python train_C_sr_S_sr_kd.py --template <template_name>

ex: python train_C_sr_S_sr_kd.py --template FusionNet_7_2s_1

if C branch checkpoint is already available, you can skip training C with the option "--skip-C"

ex: python train_C_sr_S_sr_kd.py --template FusionNet_7_2s_1 --checkpoint backup/ref_model/FusionNet_7_2s_branch_0_ckpt_E_300_P_32.778.t7 --skip-C

All template names can be found in *.py in "template/"
