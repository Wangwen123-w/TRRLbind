## About TRRLbind

In this study, we present a  deep learning model, TRRLbind, which integrates a global RNA sequence channel with a local neighboring nucleotide channel to predict RNA-small molecule binding sites. This approach takes into account both sequence and structural dependencies, thereby enhancing the accuracy of predictions. In addition, TRRLbind adopts the Transformer model as the core component of its deep learning framework, and uses the self-attention mechanism to capture the complex relationships between different positions in the input sequence, thereby enhancing the feature representation ability. The experimental results show that TRRLbind has significant advantages in terms of performance.

### Requirements
- python 3.7
- transformer
- cudatoolkit 10.1.243
- cudnn 7.6.0
- pytorch 1.4.0
- numpy 1.16.4
- scikit-learn 0.21.2
- pandas 0.24.2

The easiest way to install the required packages is to create environment with GPU-enabled version:
```bash
conda env create -f environment.yml
conda activate TRRLbind_env
```

### Testing the model

```bash
cd ./src/
python predict.py
```
### Re-training your own model for the new dataset
```bash
cd ./src/
python training.py
```
### contact
Wenjun Wang：2087707058@qq.com
