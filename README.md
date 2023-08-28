# syntorch
 sr and mask training on pytorch
 
Ref models:
[https://drive.google.com/drive/folders/1uSCfpc1W3IppaoU-NM9bZtVnT3E94oZK?usp=sharing](https://drive.google.com/drive/folders/1uSCfpc1W3IppaoU-NM9bZtVnT3E94oZK?usp=sharing)

Dataset:
[https://drive.google.com/drive/folders/10UEK2L5AxIZUD3Cw1FKPLOXUi4uHIHoV?usp=sharing](https://drive.google.com/drive/folders/10UEK2L5AxIZUD3Cw1FKPLOXUi4uHIHoV?usp=sharing)

## Commands:

### Testing
#### For test FusionNet_7_2s_1
python test_random_gradso.py --template FusionNet_7_2s_1 --checkpoint <model_name>

#### For test FusionSM_7_4s
```bash
python test_smsr.py --core FusionSM_7_4s --testset_dir [path/to/testset]
```

### Training
For training C branch with SR and S branch with KD loss:

python train_C_sr_S_kd.py --template <template_name>

ex: python train_C_sr_S_kd.py --template FusionNet_7_2s_1

for training C branch with SR and S branch with SR+KD loss:

python train_C_sr_S_sr_kd.py --template <template_name>

ex: python train_C_sr_S_sr_kd.py --template FusionNet_7_2s_1

if C branch checkpoint is already available, you can skip training C with the option "--skip-C"

ex: python train_C_sr_S_sr_kd.py --template FusionNet_7_2s_1 --checkpoint backup/ref_model/FusionNet_7_2s_branch_0_ckpt_E_300_P_32.778.t7 --skip-C

All template names can be found in *.py in "template/"
