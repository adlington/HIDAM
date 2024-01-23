# HIDAM

Source code for HIDAM.

## Requirements

* python >= 3.6
* pytorch >= 1.8.1
* dgl >= 0.6.1
* pandas >= 1.2.3
* scikit-learn >= 0.24.1

## Usage

* Create a new folder `data` under the project path:
```
cd HIDAM
mkdir data
```

* Download DBLP data from [here](https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX?usp=sharing) to `data`, where the dataset has already been preprocessed in [Heterogeneous Graph Benchmark(HGB)](https://github.com/THUDM/HGB)
```
cd data
unzip DBLP.zip
```

* Run HIDAM on the public dataset DBLP:
```
# cpu training
python main.py
```
or
```
# gpu training
python main.py --cuda
```
