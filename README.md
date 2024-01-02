# Overview
GEnDDn: Predicting lncRNA-disease associations by combining non-negative matrix factorization, graph attention autoencoder, deep neural network, and dual-net neural architecture

![chart](chart.png)
# Data
In this work，lncRNADisease is data1 and MNDR is data2.
# Environment
Install python3.6 for running this model. And these packages should be satisfied:
* numpy = 1.19.5
* pandas = 1.1.5
* tensorflow-gpu = 2.6.0
# Usage
Default is 5-fold cross validation from $C V_{l}$ to $C V_{\text {ind }}$ on lncRNADisease and MNDR databases. To run this model：
```Java
python mian.py
 ```
Default is 5-fold cross validation from $C V_{l}$ to $C V_{\text {ind }}$ on lncRNADisease and MNDR databases. To run this model：
```Java
python mian.py
 ```
Extracting linear and nonlinear features for diseases and lncRNAs to run:
```Java
python feature.py
 ```
