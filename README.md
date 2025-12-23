# Détection de Fissures sur des murs par Transfer Learning (ResNet-34)

L'objectif est d'effectuer une classification binaire pour détecter la présence de fissures sur des murs d'habitations, en utilisant des techniques d'apprentissage profond et de transfert de connaissances (Transfer Learning).

## Contexte du Projet

La détection de défauts structurels, comme les fissures, est un enjeu majeur. Ce projet se concentre sur une classification binaire : **Fissure (Crack)** vs **Pas de fissure (Non-Crack)**.

La difficulté principale résidait dans la nature du jeu de données de test (murs d'habitations avec architectures complexes) et l'absence de données d'entraînement parfaitement correspondantes. Nous avons dû faire preuve de créativité en utilisant un dataset de fissures au sol pour affiner (fine-tune) notre modèle.

## Architecture et Méthode

Plutôt que d'entraîner un modèle à partir de zéro (ce qui a donné de mauvais résultats lors de nos tests préliminaires sur des CNN standards), nous avons opté pour le **Transfer Learning** avec **ResNet-34**.

### Pourquoi ResNet-34 ?
*   **Pré-entraînement :** Le modèle a été pré-entraîné sur ImageNet1K, lui permettant de reconnaître des formes géométriques de base (coins, lignes) dès le départ.
*   **Architecture Résiduelle :** Permet une grande stabilité d'apprentissage et évite la dégradation des performances sur les réseaux profonds.
*   **Adaptation :** La dernière couche "fully connected" a été modifiée pour prédire 2 classes au lieu de 1000.

### Configuration Technique
Le modèle a été implémenté avec **PyTorch**. Voici les hyper-paramètres retenus qui ont offert la meilleure généralisation :

*   **Optimiseur :** AdamW (meilleure régularisation que SGD ou Adam classique)
*   **Fonction de coût :** Cross Entropy Loss
*   **Learning Rate :** 0.001
*   **Weight Decay :** 0.0001
*   **Batch Size :** 128
*   **Image Input :** Redimensionnement en 224x224 avec normalisation

## Résultats

Le modèle atteint une **précision globale de 76%** sur le jeu de données de test.

*   **Précision sur les fissures (Crack) :** ~84% (Le transfert de connaissances a très bien fonctionné pour cette classe).
*   **Précision sur les non-fissures (Non-Crack) :** ~14% (Le modèle peine à généraliser sur les textures de murs sains qu'il n'a pas vu lors de l'entraînement).

## Matériel utilisé

Les expérimentations ont été réalisées sur la configuration suivante :
*   **RAM :** 32 Go
*   **GPU :** Nvidia P100 (16 Go)

## Contenu du dépôt

* Le notebook/script Python contenant l'implémentation du modèle, l'entraînement et l'évaluation.
* Le rapport final au format PDF.

## Setup

1.  Cloner le dépôt :
    git clone https://github.com/...
2.  Installer les dépendances (PyTorch, Torchvision, Ignite, Matplotlib, Seaborn).
3.  Lancer le script d'entraînement ou charger les poids du modèle pour tester sur vos propres images.

## Auteur

**ZG**
*Date : Décembre 2025*
