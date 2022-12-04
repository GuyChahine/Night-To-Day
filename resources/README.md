# Documentation

Documentation of what has been done during the project.

# 3/12/2022
---
## A la recherche de ce qui a déjà été fait dans la matière d'image enhancement

- [Night-to-Day Image Translation for Retrieval-based Localization](https://people.ee.ethz.ch/~timofter/publications/Anoosheh-ICRA-2019.pdf)

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
- Je l'ai rendu plus robuste, mais il pourrais être beaucoup plus rapid, je pourrais multiprocess/multithreade le process mais c'est assez rapide pour moi
- Je créer un script qui me permet de clean les data (inversé nuit et jour ou delete)