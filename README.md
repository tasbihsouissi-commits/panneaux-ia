# Reconnaissance des panneaux de signalisation routière

Projet de vision par ordinateur visant à détecter et reconnaître les panneaux de signalisation dans des images et des vidéos en temps réel, puis à afficher leur catégorie et leur signification au conducteur. Une attention particulière est portée à l'adaptation des résultats au contexte tunisien.

## Contexte et problématique

La détection des panneaux aide à rendre l'information routière plus accessible aux conducteurs et aux systèmes d'assistance. Les panneaux peuvent être petits, flous, partiellement masqués ou observés dans des conditions d'éclairage variables. Les jeux de données publics retenus sont allemands : leurs classes ne représentent donc pas nécessairement la signalisation tunisienne. Le projet étudie les performances et les limites de ces approches avant toute conclusion sur un usage local.

## Objectifs

- Détecter des panneaux dans des scènes routières et classifier leur type.
- Comparer une détection multiclasses de bout en bout avec une approche en deux étapes.
- Établir une référence classique HOG + SVM pour la classification.
- Évaluer les erreurs, les performances par classe et le coût d'inférence.
- Vérifier la correspondance entre les classes allemandes et la signalisation tunisienne ; documenter les données locales nécessaires à une adaptation.

## Approches envisagées

1. **YOLO11** : détection et classification conjointes, avec 43 classes ou regroupement en 4 catégories.
2. **YOLO + MobileNetV2** : YOLO détecte une boîte « panneau », puis MobileNetV2 classe la région recadrée à partir de GTSRB.
3. **HOG + SVM** : référence classique de classification à implémenter et comparer sur les mêmes données de classification.

ByteTrack est utilisé pour associer les détections entre les images d'une vidéo et stabiliser l'affichage par piste. L'application encadre les panneaux et affiche leur classe, leur confiance et leur signification ; une alerte visuelle ou sonore est prévue pour certains panneaux. Le suivi est une démonstration applicative ; il ne remplace pas l'évaluation de détection sur un jeu de test annoté.

## Jeux de données

| Jeu | Tâche | Contenu et utilisation | Source |
|---|---|---|---|
| [GTSDB](https://benchmark.ini.rub.de/gtsdb_dataset.html) | Détection | 900 images de scènes routières (1360 × 800) et annotations de panneaux. Les scripts préparent les divisions train/validation/test à partir des séquences disponibles. | Institut für Neuroinformatik, Ruhr-Universität Bochum |
| [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) | Classification | Plus de 50 000 images d'entraînement, 43 classes et un jeu de test officiel. Les images sont organisées par classe dans `Final_Training/Images`. | Institut für Neuroinformatik, Ruhr-Universität Bochum |
| [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign) | Détection | Source complémentaire envisagée pour étudier des panneaux dans des scènes routières variées. Vérifier l'accès, les annotations et les conditions d'utilisation avant intégration. | Mapillary |

Les archives ne sont pas incluses dans Git. Consultez [`data/README.md`](data/README.md) pour le téléchargement, l'organisation locale et les précautions de licence. Les données sont allemandes ; toute évaluation tunisienne nécessitera un jeu local autorisé et annoté.

## Organisation du dépôt

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/README.md
├── docs/references.md
├── models/README.md
├── results/README.md
├── src/                 # scripts de préparation, entraînement et évaluation
└── notebooks/           # analyses exploratoires à ajouter
```

Les scripts actuels restent à la racine du projet pour préserver les commandes existantes. `src/` est réservé à leur réorganisation ultérieure.

## Installation

Python 3.10 ou 3.11 est recommandé. Une installation compatible CUDA peut accélérer l'entraînement ; choisissez la version de PyTorch adaptée à votre système depuis [la page officielle](https://pytorch.org/get-started/locally/).

```bash
cd panneaux-ia
python -m venv .venv
# Windows : .venv\Scripts\activate
# Linux/macOS : source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Préparation et exécution

Après téléchargement des archives, préparer les annotations de détection :

```bash
python prepare_gtsdb.py --src chemin/vers/FullIJCNN2013 --out datasets/gtsdb --mode 43
```

Le script accepte aussi `--mode 4` (catégories regroupées) ou `--mode 1` (une classe panneau). Vérifiez les noms de classes et les divisions générées avant l'entraînement.

```bash
python train_detect.py --data datasets/gtsdb/data.yaml --model yolo11n.pt --epochs 100 --imgsz 1280
python train_classifier.py --train-dir chemin/vers/Final_Training/Images \
  --test-dir chemin/vers/Final_Test/Images --test-csv chemin/vers/GT-final_test.csv
python evaluate.py --weights runs/detect/gtsdb_yolo11/weights/best.pt
```

Exemples de démonstration vidéo/webcam :

```bash
python app_video.py --weights best.pt --source 0
python app_video.py --weights best.pt --source route.mp4 --save sortie.mp4 --no-show
python app_video.py --weights best_1classe.pt --classifier classifier_mobilenetv2.pt --source route.mp4
```

## Prétraitement et choix méthodologiques

Les scripts convertissent les annotations GTSDB au format attendu par YOLO et organisent les images de classification GTSRB. Les inversions horizontales sont évitées car elles changent le sens de certains panneaux. Une résolution d'entraînement élevée est envisagée pour mieux représenter les petits panneaux ; elle augmente le coût mémoire. Les classes GTSRB étant déséquilibrées, l'entraînement du classifieur prévoit une pondération. Les images issues d'une même séquence doivent rester dans une seule division afin de réduire les fuites entre entraînement et validation.

## Évaluation expérimentale

- **Détection** : précision, rappel, mAP@0.5 et mAP@0.5:0.95 ; examiner aussi les résultats par classe, la taille des objets et les faux positifs.
- **Classification** : accuracy, précision/rappel/F1 macro et matrice de confusion, en particulier pour les classes peu représentées.
- **Efficacité** : latence et FPS mesurés avec le matériel, la résolution et la configuration consignés.
- Comparer les approches sur des données et divisions cohérentes. Ne pas comparer directement des scores issus de partitions différentes.

Les résultats ne sont pas encore consignés. L'entraînement complet et l'évaluation sur les archives originales restent à effectuer ; aucune performance n'est revendiquée dans cette version initiale.

## Technologies

Python, PyTorch, torchvision, Ultralytics YOLO, OpenCV, NumPy, scikit-learn et Pillow. Les dépendances sont listées dans [`requirements.txt`](requirements.txt).

## Équipe

À compléter par le groupe avant publication :

- Prénom Nom — rôle/contribution
- Prénom Nom — rôle/contribution

## État du projet

Prototype initial : scripts de préparation GTSDB, entraînement/démonstration YOLO et classification MobileNetV2 présents. La baseline HOG + SVM, les expériences comparatives, les résultats et l'adaptation tunisienne restent à réaliser ou documenter. Voir [`docs/references.md`](docs/references.md) pour les sources.
