# Partie 1 : Système de reconnaissance de parole en utilisant coefficients LPC et une reconnaissance basée sur l'algorithme k-NN

## Technologies Utilisées
- Python 3.9.12
- Librairies : NumPy, Librosa, Matplotlib, SciPy

Ce projet implémente un système simple de reconnaissance de mots isolés en anglais. Le système utilise une paramétrisation du signal de parole à l'aide des coefficients LPC (Linear Predictive Coding) et une reconnaissance basée sur l'algorithme des k plus proches voisins (k-NN), en utilisant la distance élastique (Dynamic Time Warping - DTW).

### Étape 1 : Lecture des Fichiers Audio
Les fichiers audio sont lus et stockés à partir d'un dossier spécifié. Chaque fichier audio correspond à un chiffre prononcé en anglais par différentes personnes.

### Étape 2 : Calcul des Coefficients LPC
Les coefficients LPC sont calculés pour chaque fichier audio pour paramétrer les signaux de parole. Ces coefficients servent de caractéristiques pour la reconnaissance de la parole.

### Étape 3 : Calcul de la Matrice des Distances DTW
Une matrice de distances DTW est calculée pour toutes les paires de séquences LPC. Cette matrice mesure la distance élastique entre chaque paire de signaux, en tenant compte des variations de vitesse et de prononciation.

### Étape 4 : Séparation des Données en Ensembles d'Entraînement et de Test
Les données sont divisées en ensembles d'entraînement et de test. Les étiquettes correspondantes sont également préparées pour chaque ensemble.

### Étape 5 : Classification k-NN
L'algorithme des k plus proches voisins est utilisé pour classer les données de test en fonction de leur distance aux données d'entraînement. La classification est réalisée pour différentes valeurs de k, et les performances sont évaluées en termes d'exactitude, de rappel et de score F1.

### Étape 6 : Visualisation des Performances
Les performances du modèle pour différentes valeurs de k sont visualisées à l'aide de graphiques, montrant l'exactitude, le score F1 et le rappel.


## Conclusion de la partie 1
Cette première partie du projet démontre la mise en place d'un système de base pour la reconnaissance de mots isolés en utilisant une méthode à l'aide des coefficients LPC (Linear Predictive Coding) et une reconnaissance basée sur l'algorithme des k plus proches voisins (k-NN). En résumé les résultats trouvées:

- Exactitude (0.34) : Seulement environ 34% des prédictions sont correctes. Cela indique une faible performance globale du modèle.
- Rappel (0.34) : Le modèle détecte correctement environ 34% des instances positives de chaque classe. Un faible rappel indique que de nombreux cas positifs réels ne sont pas correctement identifiés par le modèle.
- Score F1 (0.33) : Ce score, qui équilibre la précision et le rappel, est également bas. Un score F1 faible suggère que le modèle n'est ni précis ni complet dans ses prédictions.

Nous avons aussi tenter d'utiliser la validation croisée, les résultats indique:

Stabilité du Modèle : La stabilité des scores d'exactitude à travers les différents folds suggère que le modèle n'est pas excessivement sensible aux variations spécifiques des données dans chaque fold. Cela indique une certaine robustesse.

Performance Modérée : Une exactitude d'environ 33% est assez modeste pour un système de classification. Cela peut indiquer que les caractéristiques actuelles (coefficients LPC avec l'ordre choisi) et/ou la méthode de classification (k-NN avec la distance DTW) ne capturent pas entièrement les nuances nécessaires pour une classification plus précise.

# Partie 2 : Système de Reconnaissance de la Parole avec MFCC et HMM

## Description
Cette partie du projet vise à développer un système de reconnaissance de la parole en utilisant les coefficients MFCC (Mel-Frequency Cepstral Coefficients) pour la paramétrisation du signal audio et les Modèles de Markov Cachés (HMM) pour la classification. 


## Technologies Utilisées
- Python 3.9.12
- Librairies : NumPy, Librosa, hmmlearn, Scikit-learn

## Processus

### Étape 1 : Extraction des Coefficients MFCC
Les MFCC sont extraits de chaque fichier audio en utilisant la bibliothèque `librosa`. Ces coefficients fournissent une représentation compacte du signal vocal.

```python
def extract_mfcc(audio_file, n_mfcc=15):
    audio, Fe = librosa.load(audio_file)
    mfcc_features = mfcc(y=audio, sr=Fe, n_mfcc=n_mfcc, win_length=512, hop_length=512//2)
    return mfcc_features.T
```

### Étape 2 : Préparation des Données pour les HMM
Les données MFCC sont préparées pour l'entraînement des modèles HMM. Chaque fichier audio est traité pour obtenir une séquence de coefficients MFCC, qui sont ensuite agencés dans une matrice pour l'ensemble du dataset.

```python
def prepare_data_for_hmm(audio_files, n_mfcc=15):
    # ... (code pour préparer les données MFCC) ...
```

### Étape 3 : Entraînement des Modèles HMM
Un modèle HMM est entraîné pour chaque classe (chiffre) du dataset. La bibliothèque `hmmlearn` est utilisée pour créer et entraîner ces modèles.

```python
def train_hmm_models(audio_files_by_class):
    # ... (code pour entraîner les modèles HMM) ...
```

### Étape 4 : Division du Jeu de Données
Le jeu de données est divisé en ensembles d'entraînement et de test en s'assurant que chaque classe est représentée équitablement dans les deux ensembles.

```python
def split_dataset_by_class(dataset, test_size=0.2):
    # ... (code pour diviser le dataset) ...
```

### Étape 5 : Reconnaissance et Classification
Les modèles HMM entraînés sont utilisés pour classer de nouveaux échantillons audio en comparant les scores de vraisemblance.

```python
def classify_audio(models, test_audio_file):
    # ... (code pour classer un échantillon audio) ...
```

### Évaluation du Modèle
Le modèle est évalué sur l'ensemble de données de test en utilisant des métriques telles que l'exactitude, le rappel et le score F1.

```python
def evaluate_model(models: dict, test_features: list, true_labels: list) -> tuple[float, float, float]:
    """
    Évalue le modèle en utilisant les caractéristiques MFCC des données de test.

    :param models: Dictionnaire des modèles HMM entraînés.
    :param test_features: Liste des caractéristiques MFCC pour les données de test.
    :param true_labels: Liste des vraies étiquettes pour les données de test.
    :return: Tuple contenant l'exactitude, le rappel et le score F1.
    """
```

## Conclusion

Cette méthode emploie des technologies sophistiquées de traitement du signal et d'intelligence artificielle pour élaborer un système performant et solide de reconnaissance vocale en se basant sur les coefficients MFCC. Pour récapituler, les résultats obtenus grâce à cette technique :

- Exactitude (Accuracy) : 0,6833 (68,33 %) - Cela signifie que 68,33 % des prédictions du modèle étaient correctes. En d'autres termes, le modèle a correctement identifié la classe des fichiers audio plus des deux tiers du temps, ce qui est une performance raisonnablement bonne.

- Rappel (Recall) : 0,6833 (68,33 %) - Le rappel est également de 68,33 %, ce qui indique que le modèle est capable d'identifier correctement 68,33 % de toutes les instances positives à travers les classes.

- Score F1 : 0,6874 (68,74 %) - Le score F1, qui équilibre la précision et le rappel, est de 68,74 %. C'est un bon indicateur que le modèle a une performance équilibrée en termes de précision et de rappel.
