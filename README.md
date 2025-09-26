# DASHA

Original PyTorch implementation of DASHA from the paper ["Specialized Foundation Models Struggle to Beat Supervised Baselines"](https://arxiv.org/abs/2411.02796)
Implementation for **AutoAR** can be found [here](https://github.com/Zongzhe-Xu/AutoAR)

## Installation
To run the code, install the dependencies:

```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 torchtext==0.16.0 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install scipy tqdm ml-collections h5py requests tifffile
pip install ray[tune] datasets rasterio pytorch-lightning matplotlib
cd relax && pip install -e .
```
or run the following command:
```bash
pip install -r requirements.txt
cd relax && pip install -e .
```

## Datasets
### Genomics
We use the Nucleotide Transformer Benchmark datasets from [Hugging Face](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks) for our experiments. The datasets are automatically downloaded when running the code.

### Time Series Forecasting
Raw data consists of csv files where each file corresponds to a single dataset, the columns are channels, and the rows are timesteps. The contents should include

- ETT-small datasets (4 of them): ETTh1, ETTh2, ETTm1, ETTm2
- Electricity
- Illness
- Traffic
- Weather

**Setup**:
- Download the `all_six_dataset.zip` file from [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
- After downloading the data, put the folder `all_six_dataset/` under `./src/data/ts_datasets/`

## Reproduce results from the paper
To reproduce the results from the paper, run the following commands:
```bash
cd src && python main_dasha.py --save_dir [save_dir] --dataset [dataset]
```
or use the `run.sh` script:
```bash
chmod +x src/run.sh
cd src && ./run.sh [dataset]
```
- **For Time Series datasets,** the `--dataset` argument should be of format `[dataset]_[horizon]`, where `horizon` is the forecasting horizon. For example, to run the ETTh1 dataset with a horizon of 96, the `--dataset` argument should be `ETTh1_96`.

- To select a particular backbone architecture, use the `--arch` argument.

## Experiment with your own datasets
### Preparation
1. Add dataloaders to `./src/data_loaders.py` and complete the `get_data` function in `./src/task_configs.py`
2. Add your customized loss functions and evaluation metrics to `./src/task_utils.py` and complete the get_metric function in `./src/task_configs.py`.
3. Modify the `get_config` function in `./src/task_configs,py` according to your task.

### Run DASHA
```bash
cd src && python main_dasha.py --save_dir [save_dir] --dataset [dataset] --arch [arch]
```

## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{
  xu2025specializedfoundationmodelsstruggle,
  title={Specialized Foundation Models Struggle to Beat Supervised Baselines},
  author={Zongzhe Xu and Ritvik Gupta and Wenduo Cheng and Alexander Shen and Junhong Shen and Ameet Talwalkar and Mikhail Khodak},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=JYTQ6ELUVO}
}
```
