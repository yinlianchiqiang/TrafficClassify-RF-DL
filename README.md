# TC

## 套件

1. 安裝 requirements

```
pip install -r requirements.txt
```

2. 產生 requirements

```
pip freeze > requirements.txt
```

## 常用操作

- 開啟虛擬環境
  .\venv\Scripts\activate

## 指令 (單封包)

1. 預處理

```js
py -m 01_preprocess.p_single_packet
```

## 指令 (多封包)

1. 預處理 (基本處理 + 從封包取出需要用的欄位)

```
py -m 01_preprocess.p_graphlet_1
```

2.

```
py -m 01_preprocess.p_graphlet_2
```

## 指令

1. 切分資料集

```js
py -m 02_splitDataset.split_dataset
```

2. 集中式學習

- CNN

```js
py -m 03_train.t_cnn_main_1d
```

- RF

```js
py -m 03_train.t_rf
```

3. AE

```
py -m 03_autoencoder.ae
```

4. FL

- CNN

```js
py -m 04_train_fl.server
py -m 04_train_fl.client -i 0
py -m 04_train_fl.client -i 1
```

- RF

```js
py -m 04_train_fl.fl_rf
```
- DNN
```js
py -m 03_train.t_dnn

py -m 04_train_fl.fl_dnn_server
py -m 04_train_fl.fl_dnn_client -i 0
py -m 04_train_fl.fl_dnn_client -i 1
```

5. CAE
```js
py -m 03_autoencoder.cae

py -m 04_train_fl.fl_cae_server
py -m 04_train_fl.fl_cae_client -i 0
py -m 04_train_fl.fl_cae_client -i 1
```

6. CAE 2D Graphlet
```js
py -m 03_autoencoder.cae_2d_graphlet

py -m 04_train_fl.fl_cae_2d_graphlet_server
py -m 04_train_fl.fl_cae_2d_graphlet_client -i 0
py -m 04_train_fl.fl_cae_2d_graphlet_client -i 1
```


7. AE 2D Graphlet
```js
py -m 03_autoencoder.ae_2d_graphlet

py -m 04_train_fl.fl_ae_2d_graphlet_server
py -m 04_train_fl.fl_ae_2d_graphlet_client -i 0
py -m 04_train_fl.fl_ae_2d_graphlet_client -i 1
```

8. 2D Graphlet trans data

```
py -m 03_autoencoder.2d_graphlet_trans_data
```
