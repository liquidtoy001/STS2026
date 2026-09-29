# STS2026

[English](../../README.md) | [简体中文](README.zh-CN.md) | **Deutsch** | [Français](README.fr.md)

Ein studentisches Teamprojekt zur Untersuchung eines kompakten Sprachmodells nach dem VITS-Prinzip und seiner Evaluation.

**Status:** Eine gemeinsame Rekonstruktionsevaluation und eine lokal gemessene Baseline sind verfügbar: 2.000 Iterationen, 1.988 Optimizer-Updates. Siehe [Messprotokoll](../LOCAL_BASELINE_RESULT.md). Generalisierung, vollständige TTS-Qualität und der Vergleich mit dem neuen Modell sind noch nicht geprüft.

## Dokumentation und gemeinsame Dateien

- **[Dokumentationsseite](../site/index.html)**: Versionen (v0 Prototyp, v1 evaluierte Baseline, v2+ Verbesserungen), Architekturdiagramme, Messergebnisse und Vergleichstafel, auf Englisch, Chinesisch, Deutsch und Französisch. GitHub zeigt HTML nur als Quelltext; Repository klonen und `docs/site/index.html` im Browser öffnen.
- **[Gemeinsames Google Drive](https://drive.google.com/drive/folders/1RNj-DalkXNUA9ZGDNreAvGdHG8QWwDfU)**: Modell-Checkpoints, Laufordner und ausführliche Experimentberichte. Checkpoints und erzeugtes Audio bleiben außerhalb von Git; hier hochladen und den Ordnerpfad im PR und in `docs/site/assets/results.js` angeben.

## Einstieg

Die Teammitglieder verwenden hauptsächlich **macOS und die GitHub-Weboberfläche**. Einladungen annehmen, Dokumentation bearbeiten und Pull Requests (PRs) prüfen ist ohne Git-Installation möglich.

1. Sende deinen GitHub-Benutzernamen oder Profillink an die projektverantwortliche Person und nimm die Einladung mit demselben Konto an.
2. Lies das [GitHub-Teamhandbuch](../guides/STS2026_GitHub_Team_Handbook.pdf): Seite 5 erklärt die Annahme von Einladungen, Seiten 6–9 den Web-Workflow und Seite 19 eine Einstiegsübung vollständig im Browser.
3. Erstelle für jede Aufgabe einen Branch von `main`, committe deine Änderungen in diesen Branch und öffne einen PR mit dem Ziel `main`.
4. Bitte ein anderes Teammitglied um Review. Zu Beginn koordiniert die projektverantwortliche Person die Merges.

Ein Commit über die Website ist bereits auf GitHub gespeichert; ein zusätzlicher Push ist nicht nötig. Zum Ausführen von Code benötigst du die unter [Einrichtung](../SETUP.md) beschriebene Umgebung. Die ergänzende Dokumentation und die PDFs sind derzeit überwiegend auf Englisch.

## Aktueller Stand des Notebooks

[`vits-light_v1.ipynb`](../../notebooks/vits-light_v1.ipynb) enthält Text- und Posterior-Encoder, ein Flow-Modell, Alignment- und Dauerkomponenten, einen Wellenform-Decoder, Verlustfunktionen und einen Trainer. Die **aktive Versuchskonfiguration** deckt weniger ab als die gesamte Architektur:

- LJSpeech mit einem fest eingetragenen Kaggle-Datenpfad und Audio mit 22.050 Hz.
- 2.000 Overfit-Schritte mit acht festen LJSpeech-Beispielen.
- Dauer- und KL-Verluste haben das Gewicht null; rekonstruktionsorientierte Verluste sind aktiv.
- Der GAN-Verlust ist weiterhin null; der Diskriminator ist ein Platzhalter.
- Gemeinsame Rekonstruktionsauswertung mit festen Segmenten, Metriken und Audioexport; siehe [Evaluationsanleitung](../EVALUATION.md).
- CUDA wird verwendet, wenn verfügbar, sonst die CPU; Apples MPS-Backend wird nicht ausgewählt.

Die Messungen betreffen die Rekonstruktion von Trainingsaudio, keine vollständige VITS-Implementierung oder validierte Stimmkonvertierung. Das Notebook enthält keine gespeicherten Zellausgaben. Lies vor der Ausführung die [Einschränkungen der Baseline](../SETUP.md#current-limitations).

## Repository-Struktur

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

`tmp/`, Laufzeitausgaben in `output/`, Daten, Umgebungen, Caches, Checkpoints und erzeugte Audiobeispiele bleiben lokal und werden ignoriert. Ausgewählte PDFs liegen in `docs/guides/`; weitere Berichtsmaterialien werden vor der Aufnahme in die Dokumentation geprüft.

## Vereinbarungen im Team

- Ein kurzlebiger Branch pro Aufgabe; `main` enthält den geprüften Stand.
- Zuständigkeiten vor Notebook-Änderungen absprechen. Gleichzeitiges Ersetzen kompletter Notebooks per Web-Upload vermeiden.
- In jedem PR genau angeben, was geprüft wurde und was ungeprüft bleibt.
- Daten, Modellgewichte und Zugangsdaten nicht in Git aufnehmen; auch Notebook-Ausgaben prüfen.
- Branch-Schutz auf GitHub wird nicht vorausgesetzt. Die Teamregeln gelten auch, wenn die Oberfläche direkte Commits erlaubt.

Siehe [Beitragsrichtlinien](../../CONTRIBUTING.md) und [Dokumentation](../README.md).

## Validierung und Weiterverwendung

Elf Verhaltens-, Berichts- und Exporttests sowie Notebook-Schemaprüfungen wurden bestanden. Zusätzlich wurde die ursprüngliche Modellkonfiguration lokal mit echten Sprachdaten ausgeführt. Generalisierung, plattformübergreifende Reproduzierbarkeit und Hörqualität bleiben ungeprüft. Siehe [Validierungsprotokoll](../EVALUATION_VALIDATION.md).

Es wurde noch keine Softwarelizenz gewählt. Vor Veröffentlichung oder Weiterverwendung sind Kursvorgaben, Codeherkunft sowie Bedingungen für Drittinhalte und Daten zu prüfen. Über eine Lizenz entscheidet das Team gemeinsam.
