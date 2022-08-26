# syntorch
 sr and mask training on pytorch
 
&nbsp; | psnr | psnr | train | train | train | train | params | params
--- | --- | --- | --- |--- |--- | --- | --- | ---
&nbsp; | set14 | set5 | strat | solver | data | regularization (weight decay) | torch | matlab
[IDAG_M1_c3](/model/IDAG_M1_c3.py) | 32.883 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/1mevsb5qidSimxufvIz2M7cgfpFL47Q0j/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1-SE4qxzL3zyjnmWTv294dO6MEyDwBBEc/view?usp=sharing)
[IDAG_M1_l32](/model/IDAG_M1_l32.py) | 32.865 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/1T8w0E2JmJoJ34v6O7ZQk85AKXuVQs08Q/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1BqGvbeM3kaiTj35TG1vKhxO-M_CV4Lq6/view?usp=sharing)
[IDAG_M1_l64](/model/IDAG_M1_l64.py) | 32.947 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/1L6FOSwsksjGjRG-JkhnGBiQ7QS9IHxUU/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1jTTvo6IzTIaZmZX8tvvWSEzd0rlX9ZGk/view?usp=sharing)
[IDAG_M1_r3](/model/IDAG_M1_r3.py) | 32.922 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/1DA_6kBjEAlisBHMYLmkOEMgY9NIN7SNL/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1QO47QAwFkorjLjkh-p96SdYokywmjgLs/view?usp=sharing)
[IDAG_M2](/model/IDAG_M2.py) | 32.959 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/16x3rLtHZxpBQfZiaaOsVdezFZkPGD5Jx/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1Z2QOkTBj63ALipANezhjfJlcLZ8wQPyi/view?usp=sharing)
[IDAG_M3](/model/IDAG_M3.py) | 33.106 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/1Zp_S_BitcdZ79X5Vz9O1ZaG69M-8Env8/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/16jlFWgT5moxbGbz8H7-0PfqoLg15m4wT/view?usp=sharing)
[IDAG_M3_KD2](/model/IDAG_M3_KD2.py) | 32.765 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/1ESvXYT9_gbtkCwMr_Y9iNZN6_hOSc7mL/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/18CbRZ3jlvOeboUfBF6KgcJFafTdtkoLd/view?usp=sharing)
[IDAG_M3_KD3](/model/IDAG_M3_KD3.py) | 32.987 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/1u6fQrLigJa94nsnOrMzqYn1hfE4MmQvZ/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1eXjQtl7sneyR0ozA7hlzy-kq4LLOIxDJ/view?usp=sharing)
[IDAG_M3_KD3s](/model/IDAG_M3_KD3s.py) | 32.904 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/1ez98qWkqc77CC5ZklRbGfX1XbzLyaJvs/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1H6xdThBOfcW9wPC4Sgd5xfpzw1WoPFOT/view?usp=sharing)
[IDAG_M4](/model/IDAG_M4.py) | 32.966 | &nbsp; | normal | adam | 291 | 0 | [GoogleDrive](https://drive.google.com/file/d/1yd5fqAqqIFpdo2AznhI6aWGhgesqdT6e/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1ZJs9OE9HZXdTG0o6DCen40UkRiUkgjYE/view?usp=sharing)
[IDAG_M4](/model/IDAG_M4.py) | 32.976 | &nbsp; | finetune | adam | div2k | 0 | - | -
[IDAG_M4](/model/IDAG_M4.py) | 32.449 | &nbsp; | normal | adam | 291 | L2 (1e-4) | - | -
[IDAG_M4](/model/IDAG_M4.py) | 32.862 | &nbsp; | normal | adam | 291 | L2 (1e-5) | - | -
[IDAG_M4](/model/IDAG_M4.py) | 32.984 | &nbsp; | normal | adam | 291 | L2 (1e-6) | [GoogleDrive](https://drive.google.com/file/d/1VGVIqmZuiOjbQ_W4XegzfRdLRchJeOH7/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1KACqJ1pYr1B_nONHffftorx5znXp3zRr/view?usp=sharing)
[IDAG_M2](/model/IDAG_M2.py) | 33.029 | &nbsp; | normal | adam | 291 | L2 (1e-6) | [GoogleDrive](https://drive.google.com/file/d/1G8djcxq8ua1U5-axtMFI9qnA11TGW7P7/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1yF7qjWd-wzTT2ox4cndmXOrkmphcjITU/view?usp=sharing)
[IDAG_M6](/model/IDAG_M6.py) | 33.029 | &nbsp; | normal | adam | 291 | L2 (1e-6) | [GoogleDrive](https://drive.google.com/file/d/1GQuCxUyy9l4uoW7O2O9WduQULksRjtIr/view?usp=sharing) | [GoogleDrive](https://drive.google.com/file/d/1msvYXfYfykCVgsXmHNoVQvxe7l-rDnIs/view?usp=sharing)
