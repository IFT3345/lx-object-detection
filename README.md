<p align="center">
<img src="./assets/images/dtlogo.png" alt="Duckietown Logo" width="50%">
</p>

# Labo 6: Détection d'objets 

Ce laboratoire vous guidera à travers le processus de collecte et d'annotation automatique des données,
et leur utilisation pour entraîner un réseau neuronal à détecter les objets à partir de l'image de la caméra du robot. Nous utiliserons ensuite ce modèle entraîné
pour éviter d'écraser les canards piétons à Duckietown.
Nous utiliserons l'un des réseaux neuronaux de détection d'objets les plus populaires, appelé [YOLO (v11)](https://docs.ultralytics.com/models/yolo11/).
Vous devrez également intégrer ce modèle entraîné à un contrôleur à retour d'information afin d'éviter d'écraser les canards.
Pour l'instant, nous nous arrêterons simplement dès qu'un objet (un canard) sera détecté sur la route.


##  Mais d'abord...

Assurez-vous que votre système est à jour.

- 💻 Veillez toujours à ce que votre Duckietown Shell soit mise à jour vers la dernière version: 

```
     pipx upgrade duckietown-shell
```

- 💻 Mettre à jour les commandes du shell: 

```
     dts update
```

- 💻 Assurez-vous que toutes les images Docker présentes sur votre ordinateur sont à jour: 

```
     dts desktop update
```

- 💻 Arrêtez et supprimez tous les conteneurs Docker existants (que l'autre groupe aurait pu laisser ouverts par erreur):


```
     docker stop $(docker ps -aq)
     docker rm $(docker ps -aq)
```     

- 💻 Vous devrez peut-être également supprimer ce répertoire temporaire pour avoir les autorisations nécessaires pour y écrire.

```
     sudo rm -rf /tmp/duckiematrix
```
   
- 🚙 Assurez-vous que toutes les images Docker présentes sur votre ordinateur sont à jour: 

```
    dts duckiebot update ROBOTNAME
```

(où ROBOTNAME est le nom de votre Duckiebot — réel ou virtuel. Assurez vous que ce dernier soit actif.)



## Avant de commencer


**REMARQUE 1** : Vous aurez également besoin d’un compte [Hugging Face](https://huggingface.co). Cliquez sur le bouton « Sign Up » en haut à droite pour créer un compte.

**REMARQUE 2** : Pour utiliser le modèle SAM3, vous devrez demander l’accès en [remplissant le formulaire de demande](https://huggingface.co/facebook/sam3). L’approbation peut prendre quelques minutes ; si vous faites votre demande maintenant, elle sera approuvée avant même que vous passiez à l’étape de l’étiquetage automatique.




# Comment réaliser cet exercice de laboratoire ?

## Lancez l'éditeur de code.

Ouvrez l'éditeur de code (VSCode) en exécutant la commande suivante:

```
dts code editor --gpus all
```

**REMARQUE** L'option `--gpus all` rendra votre GPU disponible dans l'éditeur VSCode.


Attendez qu'une URL s'affiche dans le terminal, puis cliquez dessus ou copiez-la et collez-la dans la barre d'adresse de votre navigateur pour accéder à l'éditeur de code. Le premier élément que vous verrez dans l'éditeur de code est ce même document. 

**Vous pouvez poursuivre votre travail à partir de là**


## Les notebooks "Jupyter"

**REMARQUE** : Vous devez lire ce message depuis l'éditeur de code de votre navigateur.

Dans l'éditeur de code, utilisez la barre latérale de navigation située à gauche pour accéder au
dossier `notebooks` et ouvrir le premier notebook.

Suivez les instructions du notebook et parcourez les notebooks dans l'ordre.

Une fois que vous avez terminé toutes les tâches des notebooks, vous pouvez suivre les instructions suivantes pour tester votre code.

## Exécution de votre code

### Tester avec la Duckiematrix

Il peut être utile de tester votre code dans un environnement de simulation avant de l'essayer sur le robot réel. Pour cela, nous avons la Duckiematrix.

Pour tester votre code dans Duckiematrix, vous aurez besoin d'un robot virtuel. Vous pouvez en créer un avec la commande suivante:

```
dts duckiebot virtual create [VBOT] -t duckiebot -c DB21J
```

où `[VBOT]` peut être n'importe quoi (mais n'oubliez pas ce nom pour la suite).

Vous pouvez ensuite démarrer votre robot virtuel avec la commande:

```
dts duckiebot virtual start [VBOT] --pull
```

Vous devriez le voir avec le statut « Booting » (démarrage) et enfin « Ready » (prêt) si vous consultez la commande `dts fleet discover` :

```
     | Hardware |   Type    | Model |  Status  | Hostname 
---  | -------- | --------- | ----- | -------- | ---------
[VBOT] |  virtual | duckiebot | DB21J |  Ready   | [VBOT].local
```

Maintenant que votre robot virtuel est prêt, vous pouvez démarrer Duckiematrix. Depuis ce répertoire d'exercices, exécutez la commande suivante :

```
dts code start_matrix
```

Vous devriez voir le simulateur Duckiematrix, basé sur Unity, démarrer. L'écran de démarrage ressemblera à ceci :

![duckiematrix_start](assets/images/duckiematrix-start.png)

À partir d'ici, vous pouvez cliquer n'importe où dans la fenêtre et appuyer sur la touche [ENTRÉE] pour l'activer. Vous pouvez ensuite déplacer le petit canard vers le Duckiebot à l'aide des touches « w », « a », « s » et « d », ou modifier l'angle de la caméra pour observer le Duckiebot avec la souris. Vous pouvez également passer à une vue de dessus en appuyant sur la touche « v », ce qui vous donnera une vue similaire à celle-ci :

![duckiematrix_overhead](assets/images/duckiematrix-overhead.png)




### "Build" votre code

Vous pouvez build le code avec

```
dts code build -R ROBOTNAME
```

où ROBOTNAME peut être un robot réel ou virtuel.

### Tester le code

Vous pouvez ensuite exécuter votre code avec

```
dts code workbench -R ROBOTNAME [-m]
```

où ROBOTNAME peut être un robot réel ou virtuel, mais s'il s'agit d'un robot virtuel, vous devez inclure l'option `-m` pour indiquer que vous souhaitez le tester dans la Duckiematrix.




Cependant, avant de pouvoir effectuer des tests, vous devrez :

- Collecter les données
- Annoter ces données (automatiquement)
- Entraîner votre modèle de détection d'objets
- Exporter votre modèle


Pour commencer, vous pouvez passer au [premier notebook qui est une introduction aux réseaux neuronaux et aux CNN](./notebooks/01-CNN/intro_to_nn.ipynb).

## Credits

La version précédente (daffy) de ce LX a été en grande partie écrite par [Charlie Gauthier](https://velythyl.github.io/).

Cette version mise à jour (ente) a été en grande partie écrite par [Shima Shahfar](https://ca.linkedin.com/in/shima-shahfar).
