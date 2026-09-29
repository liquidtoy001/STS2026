# STS2026

[English](../../README.md) | [简体中文](README.zh-CN.md) | [Deutsch](README.de.md) | **Français**

Projet de cours en équipe consacré à un modèle vocal compact inspiré de VITS et à son évaluation.

**État actuel :** une évaluation de reconstruction partagée et une référence mesurée localement sont disponibles : 2 000 itérations, 1 988 mises à jour de l’optimiseur. Voir le [compte rendu](../LOCAL_BASELINE_RESULT.md). La généralisation, la qualité TTS complète et la comparaison avec le nouveau modèle restent à vérifier.

## Documentation et fichiers partagés

- **[Site de documentation](../site/index.html)** : versions (prototype v0, référence évaluée v1, améliorations v2+), schémas d’architecture, résultats mesurés et tableau comparatif, en anglais, chinois, allemand et français. GitHub affiche le HTML comme du code source : clonez le dépôt et ouvrez `docs/site/index.html` dans un navigateur.
- **[Google Drive partagé](https://drive.google.com/drive/folders/1RNj-DalkXNUA9ZGDNreAvGdHG8QWwDfU)** : checkpoints du modèle, dossiers d’exécution et rapports d’expérience détaillés. Les checkpoints et l’audio généré restent hors de Git ; déposez-les ici et indiquez le chemin du dossier dans la PR et dans `docs/site/assets/results.js`.

## Pour commencer

Les membres de l’équipe utilisent principalement **macOS et le site web GitHub**. Il est possible d’accepter une invitation, de modifier la documentation et de relire des pull requests (PR) sans installer Git.

1. Transmettez votre nom d’utilisateur GitHub ou le lien de votre profil à la personne responsable du dépôt, puis acceptez l’invitation avec ce même compte.
2. Consultez le [guide GitHub de l’équipe](../guides/STS2026_GitHub_Team_Handbook.pdf) : la page 5 explique l’acceptation des invitations, les pages 6–9 présentent le travail sur le site web et la page 19 propose un exercice entièrement dans le navigateur.
3. Pour chaque tâche, créez une branche à partir de `main`, enregistrez vos commits sur cette branche, puis ouvrez une PR vers `main`.
4. Demandez une relecture à un autre membre. Au début, la personne responsable du dépôt coordonne les fusions.

Un commit effectué sur le site est déjà enregistré sur GitHub ; aucun push supplémentaire n’est nécessaire. Pour exécuter le code, configurez l’environnement décrit dans les [instructions d’installation](../SETUP.md). Les documents complémentaires et les PDF sont actuellement principalement en anglais.

## Fonctionnement actuel du notebook

[`vits-light_v1.ipynb`](../../notebooks/vits-light_v1.ipynb) contient des encodeurs de texte et de distribution a posteriori, un modèle de flot, des composants d’alignement et de durée, un décodeur de forme d’onde, des fonctions de perte et une boucle d’entraînement. La **configuration expérimentale active** est plus limitée que l’architecture complète :

- Utilisation de LJSpeech, avec un chemin Kaggle codé en dur et un audio à 22 050 Hz.
- 2 000 étapes de surapprentissage sur huit exemples LJSpeech fixes.
- Pondérations des pertes de durée et KL fixées à zéro ; pertes orientées reconstruction activées.
- Perte GAN toujours nulle et discriminateur à l’état d’ébauche.
- Évaluation de reconstruction partagée avec segments fixes, métriques et export audio ; voir le [guide d’évaluation](../EVALUATION.md).
- Sélection de CUDA si disponible, sinon du CPU ; le backend MPS d’Apple n’est pas sélectionné.

Les mesures portent sur la reconstruction des données d’entraînement, sans établir une implémentation VITS complète ni une conversion vocale validée. Le notebook ne contient aucune sortie de cellule enregistrée. Consultez les [limites de la version de référence](../SETUP.md#current-limitations) avant de l’exécuter.

## Structure du dépôt

```text
STS2026/
  README.md
  CONTRIBUTING.md
  .github/pull_request_template.md
  notebooks/
    vits-light_v1.ipynb
  requirements/
    notebook.txt
    docs.txt
  docs/
    README.md
    SETUP.md
    translations/
      README.zh-CN.md
      README.de.md
      README.fr.md
    guides/
      STS2026_GitHub_Team_Handbook.pdf
      ASR_VITS_Evaluation_Guide.pdf
  tools/handbook/
    build_github_handbook.py
    handbook_web_content.py
    check_github_handbook.py
  tmp/                            # Local only / Git-ignored
```

`tmp/`, les sorties d’exécution dans `output/`, les données, environnements, caches, points de contrôle et échantillons audio restent locaux et sont ignorés. Les PDF sélectionnés se trouvent dans `docs/guides/` ; les autres documents de résultats doivent être relus avant leur ajout à la documentation.

## Règles de travail

- Une branche de courte durée par tâche ; `main` contient la version relue.
- Définir les responsabilités avant de modifier un notebook. Éviter les remplacements simultanés du fichier complet par téléversement web.
- Indiquer précisément dans chaque PR ce qui a été vérifié et ce qui reste à vérifier.
- Ne pas inclure dans Git les données, poids de modèles ou identifiants secrets ; vérifier aussi les sorties des notebooks.
- Ne pas supposer que la protection des branches GitHub est activée. Respecter les règles de l’équipe même lorsque l’interface autorise un commit direct.

Voir les [règles de contribution](../../CONTRIBUTING.md) et la [documentation](../README.md).

## Validation et réutilisation

Onze tests de comportement, de rapport et d’export ainsi que les contrôles de schéma du notebook ont réussi. La configuration originale du modèle a également été exécutée localement sur de vraies données vocales. La généralisation, la reproductibilité entre machines et la qualité perceptive restent à vérifier. Voir le [compte rendu de validation](../EVALUATION_VALIDATION.md).

Aucune licence logicielle n’a encore été choisie. Avant toute publication ou réutilisation, vérifiez les règles du cours, la provenance du code et les conditions relatives aux contenus tiers et aux données. Le choix d’une licence relève de l’équipe.
