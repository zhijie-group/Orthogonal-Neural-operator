# Orthogonal Neural Operator (ONO)
Code for ONO, a neural operator built upon orthogonal attention.

## Requirements
The code depends on python 3.7

and pytorch:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```


## Datasets
- [FNO datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
- [Geo-FNO datasets](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing)


## Examples

### Darcy
```bash
python Darcy_example.py --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 10 --lr 0.001 --use_tb 1 --attn_type nystrom --max_grad_norm 0.1 --orth 1 --psi_dim 32 --batch-size 4 --mlp_ratio 2
```

### NS2d
```bash
python NS_example2.py --model ONO2 --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --use_tb 1 --max_grad_norm 0.1 --orth 1 --psi_dim 16 --batch-size 8
```

### Elasticity
```bash
python ela_example.py --model ONO2  --n-hidden 128 --attn_type nystrom  --n-heads 8 --n-layers 8 --lr 0.001 --use_tb 1 --max_grad_norm 0.1 --orth 1 --psi_dim 8 --batch-size 8
```

### Plastcity
```bash
python pla_example.py --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --use_tb 1 --max_grad_norm 0.1 --orth 1 --psi_dim 8 --batch-size 8
```

### Pipe
```bash
python pipe_example.py --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --use_tb 1 --max_grad_norm 0.1 --orth 1 --psi_dim 8 --batch-size 2 --attn_type nystrom
```

### Airfoil
```bash
python airfoil_example.py --model ONO2 --attn_type nystrom  --n-hidden 128 --n-heads 8 --n-layers 6 --lr 0.001 --use_tb 1 --max_grad_norm 0.1 --orth 1 --psi_dim 8 --batch-size 4
```

### Zero-shot Super-resolution
```bash
python space_gen.py --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --use_tb 1 --max_grad_norm 0.1 --orth 1 --psi_dim 16 --batch-size 8
```

### Time generalization on NS2d
```bash
python time_gen.py --model ONO2  --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --use_tb 1 --max_grad_norm 0.1 --orth 1 --psi_dim 8 --batch-size 8
