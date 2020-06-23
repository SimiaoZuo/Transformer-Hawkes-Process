# Transformer Hawkes Process

Source code for [Transformer Hawkes Process (ICML 2020)](https://arxiv.org/abs/2002.09291).

# Run the code

### Dependencies
1. Python 3.7.3.
2. [Anaconda](https://www.anaconda.com/) contains all the required packages.
3. [PyTorch](https://pytorch.org/) version 1.4.0.

### Instructions
1. Put the data folder inside the root folder, modify the **data** entry in **run.sh** accordingly. The datasets are available [here](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U).
2. **bash run.sh** to run the code.

### Note
* Right now the code only supports single GPU training, but an extension to support multiple GPUs should be easy.
* There are several factors that can be changed, besides the ones in **run.sh**:
  * In **transformer/Models.py**, class **Transformer**, parameter **alpha** controls the weight of the time difference factor. This parameter can be added into the computation graph, i.e., changeable during training, but the gain is marginal.
  * In **transformer/Models.py**, class **Transformer**, there is an optional recurrent layer. This  is inspired by the fact that additional recurrent layers can better capture the sequential context, as suggested in [this paper](https://arxiv.org/pdf/1904.09408.pdf). In reality, this may or may not help, depending on the dataset.
  * In **Utils.py**, function **log_likelihood**, users can select whether to use numerical integration or Monte Carlo integration.

# Reference

Please cite the following paper if you use this code.

```
@article{zuo2020transformer,
  title={Transformer Hawkes Process},
  author={Zuo, Simiao and Jiang, Haoming and Li, Zichong and Zhao, Tuo and Zha, Hongyuan},
  journal={arXiv preprint arXiv:2002.09291},
  year={2020}
}
```
