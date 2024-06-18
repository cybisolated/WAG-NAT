# WAG-NAT
The repository contains the code for paper **_WAG-NAT: Window Attention and Generator Based Non-Autoregressive Transformer for Time Series Forecasting_**, 
which is accepted by _ICANN 2023 - International Conference on Artificial Neural Networks_. 

**For your reference, our paper is uploaded in this repo as `WAG-NAT.pdf`.**

# Data
The ETT dataset can be obtained from repo [ETDataset](https://github.com/zhouhaoyi/ETDataset).
The BMSAQ dataset can be obtained from [BMSAQ](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data).

# Usage
Run `python script_get_data.py` to process the downloaded data.

Run `python script_train_${model}.py` to train the corresponding model.

Run `python script_tune_${model}.py` to tune the corresponding model.

The folder `expr/` contains all the experimental scripts to run.

# Citation
If you feel our work is interesting or this repo is helpful, you can cite our paper using the following bib:
```tex
@inproceedings{chen2023wag,
  title={WAG-NAT: Window Attention and Generator Based Non-Autoregressive Transformer for Time Series Forecasting},
  author={Chen, Yibin and Li, Yawen and Xu, Ailan and Sun, Qiang and Chen, Xiaomin and Xu, Chen},
  booktitle={International Conference on Artificial Neural Networks},
  pages={293--304},
  year={2023},
  organization={Springer}
}
```
