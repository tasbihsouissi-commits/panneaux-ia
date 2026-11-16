# Données

Les archives et images ne sont pas suivies par Git. Téléchargez-les depuis les pages officielles, lisez leurs conditions d'utilisation et conservez les fichiers d'origine sans les modifier.

## Sources

- **GTSDB (détection)** : [page officielle et téléchargement](https://benchmark.ini.rub.de/gtsdb_dataset.html). Archive attendue par les scripts : `FullIJCNN2013`.
- **GTSRB (classification)** : [page officielle et téléchargement](https://benchmark.ini.rub.de/gtsrb_dataset.html). Utiliser `Final_Training/Images`, `Final_Test/Images` et `GT-final_test.csv`.
- **Mapillary Traffic Sign Dataset (détection, optionnel)** : [page du dataset](https://www.mapillary.com/dataset/trafficsign). Évaluer son accès, le format des annotations, la couverture géographique et les conditions d'utilisation avant de décider de l'intégrer.

Consultez les pages des détenteurs pour les conditions de redistribution et de citation à jour. Ne republiez pas les images dans ce dépôt sans autorisation.

## Organisation locale proposée

```text
data/
├── raw/
│   ├── FullIJCNN2013/
│   └── GTSRB/
│       ├── Final_Training/Images/
│       ├── Final_Test/Images/
│       └── GT-final_test.csv
└── processed/             # annotations YOLO / divisions générées
```

Exemple de préparation depuis la racine `panneaux-ia` :

```bash
python prepare_gtsdb.py --src data/raw/FullIJCNN2013 --out datasets/gtsdb --mode 43
```

GTSDB comprend 900 images de scènes routières annotées ; le script répartit les images selon les indices définis dans son implémentation. GTSRB comporte 43 classes de panneaux et plus de 50 000 exemples d'entraînement. Vérifiez les comptes et le contenu des divisions générées avant de lancer les expériences. Les deux corpus représentent la signalisation allemande, pas tunisienne.
