# Night-To-Day

## CNNV1

``` python
image_shape = (64,64,3),
batch_size = 50,
res_filters = 64,
conv_filters = 16,
nb_resnet = 3,
optimizer = Adam(),
data_generator = DataGenerator((64,64), 50)
```

## CNNV2

``` python
image_shape = (64,64,3),
batch_size = 50,
res_filters = 256,
conv_filters = 16,
nb_resnet = 3,
optimizer = Adam(),
data_generator = DataGenerator((64,64), 50)
```

``` python
[Epoch 20/20] [Batch 5/5] [Loss 0.03679] [Accuracy 0.42556]
```

## CNN2V1

``` python
image_shape = (64,64,3),
batch_size = 50,
res_filters = 256,
conv_filters = 64,
nb_resnet = 9,
optimizer = Adam(),
data_generator = DataGenerator((64,64), 50)
```

``` python
[Epoch 20/20] [Batch 5/5] [Loss 0.15803] [Accuracy 0.40638]
```

``` python
[Epoch 40/40] [Batch 5/5] [Loss 0.11775] [Accuracy 0.44520]
```

``` python
[Epoch 60/60] [Batch 5/5] [Loss 0.07887] [Accuracy 0.45954]
```

``` python
[Epoch 80/80] [Batch 5/5] [Loss 0.05875] [Accuracy 0.45507]
```

``` python
[Epoch 100/100] [Batch 5/5] [Loss 0.05609] [Accuracy 0.49559]
```