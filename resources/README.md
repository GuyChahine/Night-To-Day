# Documentation

Documentation of what has been done during the project.

# 3/12/2022
---
## A la recherche de ce qui a déjà été fait dans la matière d'image enhancement

- [Night-to-Day Image Translation for Retrieval-based Localization](https://people.ee.ethz.ch/~timofter/publications/Anoosheh-ICRA-2019.pdf)
    - ![Cycle Gan Diagram](/resources/assets/CycleGanDIagram2.png)

- L'utilisation d'un Generative Adversarial Networks (GANs) peut donné de bon resultats

- Recherche sur le fonctionnement d'un GAN

![GAN Diagram](/resources/assets/Example-of-the-Generative-Adversarial-Network-Model-Architecture.png)

## Recherche sur la génération des training data

- Pour créer notre dataset nous avons donc besoin de training data composé d'image de nuit et de la meme image mais de jour

- J'essaie de transformer une photo de jour en nuit avec l'outil photoshop

1. Example :

![Day Picture](/resources/assets/france.png)


- 1er methode:
![Night Picture](/resources/assets/francenight.png)

- 2e methode:
![Night Picture](/resources/assets/francenight2.png)

2. Example :

![Night Picture](/resources/assets/people.png)

- 1er methode:
![Night Picture](/resources/assets/peoplenight.png)

## Après recherche j'ai trouvé des topics

- [How to convert night image to day image?](https://datascience.stackexchange.com/questions/31430/how-to-convert-night-image-to-day-image)

## Je cherche a trouver des timelapse qui montre le meme endroit de jour que de nuit

- J'ai trouver un site repertoriant enormement de timelaps allant du jour jusqu'a la nuit: [VIDEVO](https://www.videvo.net/)

- Je vais creer un script permettant de telecharger et extraire des photos a different moment du timelapse

![City by Day](/resources/assets/0_day.jpeg)
![City by Night](/resources/assets/0_night.jpeg)

# 4/12/2022

## Creation du dataset a partir de timelaps sur videvo

- J'ai créer un script qui recupere les mp4 url, download les video et cut la 20 premiere frame et la derniere-50 frame
- Je l'ai rendu plus robuste, mais il pourrais être beaucoup plus rapide, je pourrais multiprocess/multithreade le process mais c'est assez rapide pour moi
- Je créer un script qui me permet de clean les data (inversé nuit et jour ou delete)

![DATA CLEANER SCREENSHOT](/resources/assets/cleandata.png)

- Les 700 images de base unclean m'ont donné 200 images clean
- J'ai trouver un site pour enlever des watermark [pixelbin.io](https://www.pixelbin.io/)

- J'ai prit des video de youtube de timelaps et j'ai cuter des images nuit/jour
- J'ai reprit le dataset de 17 images je l'ai trier et upscale

## Model

- Malgré le fait qui me manque des data je vais commmencé a créer le GAN

- Je cherche sur Kaggle des information sur les GAN

- [GAN INTRODUCTION KAGGLE](https://www.kaggle.com/code/jesucristo/gan-introduction)
- [GAN with Keras: Application to Image Deblurring](https://medium.com/sicara/keras-generative-adversarial-networks-image-deblurring-45e3ab6977b5)
- [GAN by Example using Keras on Tensorflow Backend](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)
- [Video on cycle GAN](https://www.youtube.com/watch?v=42gSiS9y5Lo)
- [More paper on CycleGan](https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/)

- J'ai fini la première creation du model ne m'aidant de 2 source :
    - [github repo](https://github.com/eriklindernoren/Keras-GAN/tree/master/cyclegan)
    - [How to Develop a CycleGAN for Image-to-Image Translation with Keras](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/)

- Le reseaux GAN est très lourd et j'utilise que des image de 256x256 pixel avec des batchs de 2

# 06/12/2022

## Recherche sur different model

- [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/pdf/1703.00848.pdf)
- [Coupled Generative Adversarial Networks](https://arxiv.org/pdf/1606.07536.pdf)
- [CoGAN: Learning joint distribution with GAN](https://agustinus.kristia.de/techblog/2017/02/18/coupled_gan/)

## New CNN1

- J'ai reflechi a créer un nouveau model basé sur de CNN classique
- Le model apprend peut avec un nombre de filter bas
- Avec un nombre de filter haut il rend des images blanche (loss:0.3/0.4) donc je t'en d'InstanceNormalisé les layers

## CNN2

- J'ai InstanceNormalize toute les conv2D : Meilleur resultat le CNN descend en dessous de 0.3 -> 0.15
- [REF ON RESNET](https://arxiv.org/pdf/1512.03385.pdf)
- Je regarde les [resnet de keras](https://keras.io/api/applications/resnet/)
- Pendant le training je viens de voir une image pas clean dans mon dataset (nuit/nuit)

## Recherche model

- [Pix2PixGAN](https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/)
- Je repart sur un [CycleGan](https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py)
- Reference sur le [UNET](https://arxiv.org/pdf/1505.04597.pdf)

## CycleGAN New version

- testing sur le model, bon resultat
- Pour que le model puisse produire de meilleur resultat j'agrandi le dataset avec de nouvelle image day/night
- Aussi pour qu'il puisse reussir s'adapter plus confortablement a different situation
- [CITY DATASET DAY NIGHT KAGGLE](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)
- J'ai trier et prit 500 image pour faire un train test

- Training sur 789 pixel est très: 1h pour 6 epoch sur 500 de training data

- New dataset by [HEONH0](https://www.kaggle.com/datasets/heonh0/daynight-cityview)

- 1h40 minutes pour le run avec 789 pixels



- Je refait un run seulement sur les data de city et timelaps
- J'ai essayé de passé les donnée et tensorlfow en float16 mais le calcul des loss ne ce fait pas correctement, les valeur sont trop petit et donc egale a 0

- INFO: Le name trainv3b20128,128 avait un nb resnet a 9

## Build CycleGan with ResNet

- Idée basé sur [le code de nvidia](https://github.com/mingyuliutw/UNIT)

- ResNet produit des images floue et le Unet produit des image claire mais peut transformé contrairement au resnet

- [Combination ?](https://www.kaggle.com/code/meaninglesslives/unet-resnet34-in-keras/notebook)

## Recherche d'une loss function

- [loss function](https://arxiv.org/pdf/1511.08861.pdf)
- [Contextual Loss](https://arxiv.org/pdf/1803.02077.pdf)
- [Gram Matrix](https://github.com/robertomest/neural-style-keras/blob/master/training.py)
- [CycleGAN New one](https://github.com/simontomaskarlsson/CycleGAN-Keras/blob/05c2dab2a8346fbc8ec9b6aed06eec1a0c3d5e04/model.py#L257)

- INFO: On "trainv3_3UP_r128_nbres9" the discriminator was too good so I train it once every 2 batchs

- Je cherche a reduire le noise sur les photo CycleGANUnetv2_trainv3_3UP_r128_nbres9_e11_r(128,128)_b1
- [DN-ResNet: Efficient Deep Residual Network for Image Denoising](https://arxiv.org/pdf/1810.06766.pdf)

- The adversarial loss in a GAN represents the amount of information that the generator is able to trick the discriminator into believing is true.