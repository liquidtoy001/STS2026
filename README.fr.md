# STS2026

[English](README.md) | [简体中文](README.zh-CN.md) | [Deutsch](README.de.md) | **Français**

Projet de cours en équipe consacré à un modèle vocal compact inspiré de VITS et à son évaluation.

**État actuel :** un premier notebook de référence et des documents de planification. L’entraînement complet, la qualité de l’inférence de bout en bout et la reproductibilité de l’évaluation n’ont pas été validés lors de la préparation du dépôt.

## Pour commencer

Les membres de l’équipe utilisent principalement **macOS et le site web GitHub**. Il est possible d’accepter une invitation, de modifier la documentation et de relire des pull requests (PR) sans installer Git.

1. Transmettez votre nom d’utilisateur GitHub ou le lien de votre profil à la personne responsable du dépôt, puis acceptez l’invitation avec ce même compte.
2. Consultez le [guide GitHub de l’équipe](output/pdf/STS2026_GitHub_Team_Handbook.pdf) : la page 5 explique l’acceptation des invitations, les pages 6–9 présentent le travail sur le site web et la page 19 propose un exercice entièrement dans le navigateur.
3. Pour chaque tâche, créez une branche à partir de `main`, enregistrez vos commits sur cette branche, puis ouvrez une PR vers `main`.
4. Demandez une relecture à un autre membre. Au début, la personne responsable du dépôt coordonne les fusions.

Un commit effectué sur le site est déjà enregistré sur GitHub ; aucun push supplémentaire n’est nécessaire. Pour exécuter le code, configurez l’environnement décrit dans les [instructions d’installation](docs/SETUP.md). Les documents complémentaires et les PDF sont actuellement principalement en anglais.

## Fonctionnement actuel du notebook

[`vits-light_v1.ipynb`](vits-light_v1.ipynb) contient des encodeurs de texte et de distribution a posteriori, un modèle de flot, des composants d’alignement et de durée, un décodeur de forme d’onde, des fonctions de perte et une boucle d’entraînement. La **configuration expérimentale active** est plus limitée que l’architecture complète :

- Utilisation de LJSpeech, avec un chemin Kaggle codé en dur et un audio à 22 050 Hz.
- Appel de `overfit_train(total_steps=1500)` sur un seul lot fixe.
- Pondérations des pertes de durée et KL fixées à zéro ; pertes orientées reconstruction activées.
- Perte GAN toujours nulle et discriminateur à l’état d’ébauche.
- Fonction `evaluate(outputs, batch)` non implémentée.
- Sélection de CUDA si disponible, sinon du CPU ; le backend MPS d’Apple n’est pas sélectionné.

Il s’agit d’un point de départ pour le développement, et non de la preuve d’une implémentation VITS complète, d’une évaluation ayant produit des mesures ou d’une conversion vocale validée. Le notebook ne contient aucune sortie de cellule enregistrée. Consultez les [limites de la version de référence](docs/SETUP.md#current-limitations) avant de l’exécuter.

## Structure du dépôt

```text
STS2026/
  vits-light_v1.ipynb              Notebook de référence original
  README.md                      Page d’accueil en anglais
  README.zh-CN.md                Présentation en chinois
  README.de.md                   Présentation en allemand
  README.fr.md                   Présentation en français
  CONTRIBUTING.md                Règles de collaboration et de relecture
  requirements-notebook.txt       Dépendances du notebook hors PyTorch
  requirements-docs.txt           Dépendances de création et de contrôle des PDF
  .github/pull_request_template.md
  docs/
    SETUP.md                     Environnement, données et exécution
    README.md                    Index documentaire et instructions de génération
    build_github_handbook.py      Mise en page et génération du guide
    handbook_web_content.py       Contenu anglais du guide web
    check_github_handbook.py      Contrôle de la structure et de la mise en page
  output/pdf/
    STS2026_GitHub_Team_Handbook.pdf
    ASR_VITS_Evaluation_Guide.pdf
```

`tmp/`, les données, environnements, caches, points de contrôle et échantillons audio générés restent locaux et sont ignorés. Les deux PDF sélectionnés sont volontairement inclus. Tout autre fichier de sortie nécessite une revue et une mise à jour des règles d’exclusion.

## Règles de travail

- Une branche de courte durée par tâche ; `main` contient la version relue.
- Définir les responsabilités avant de modifier un notebook. Éviter les remplacements simultanés du fichier complet par téléversement web.
- Indiquer précisément dans chaque PR ce qui a été vérifié et ce qui reste à vérifier.
- Ne pas inclure dans Git les données, poids de modèles ou identifiants secrets ; vérifier aussi les sorties des notebooks.
- Ne pas supposer que la protection des branches GitHub est activée. Respecter les règles de l’équipe même lorsque l’interface autorise un commit direct.

Voir les [règles de contribution](CONTRIBUTING.md) et la [documentation](docs/README.md).

## Validation et réutilisation

La préparation du dépôt vérifie uniquement la structure du notebook et la syntaxe Python. Ces contrôles n’exécutent pas l’entraînement et ne prouvent pas que le modèle est correct. L’accès aux données, les versions compatibles de PyTorch/TorchAudio, un protocole d’évaluation sur des données réservées et les résultats expérimentaux restent à établir par l’équipe.

Aucune licence logicielle n’a encore été choisie. Avant toute publication ou réutilisation, vérifiez les règles du cours, la provenance du code et les conditions relatives aux contenus tiers et aux données. Le choix d’une licence relève de l’équipe.
