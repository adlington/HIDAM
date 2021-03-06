# HIDAM

Source code for paper "Heterogeneous Information Network based Default Analysis on Banking Micro and Small Enterprise Users".

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

* Download DBLP data from [here](https://cloud.tsinghua.edu.cn/d/2d965d2fc2ee41d09def/files/?p=%2FDBLP.zip&dl=1) to `data`, where the dataset has already been preprocessed in [Heterogeneous Graph Benchmark(HGB)](https://github.com/THUDM/HGB)
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