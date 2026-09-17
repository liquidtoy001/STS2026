# STS2026

[English](README.md) | [简体中文](README.zh-CN.md) | **Deutsch** | [Français](README.fr.md)

Ein studentisches Teamprojekt zur Untersuchung eines kompakten Sprachmodells nach dem VITS-Prinzip und seiner Evaluation.

**Status:** Ein erstes Baseline-Notebook und Planungsunterlagen. Vollständiges Training, die Qualität der End-to-End-Inferenz und eine reproduzierbare Evaluation wurden bei der Vorbereitung des Repositorys nicht validiert.

## Einstieg

Die Teammitglieder verwenden hauptsächlich **macOS und die GitHub-Weboberfläche**. Einladungen annehmen, Dokumentation bearbeiten und Pull Requests (PRs) prüfen ist ohne Git-Installation möglich.

1. Sende deinen GitHub-Benutzernamen oder Profillink an die projektverantwortliche Person und nimm die Einladung mit demselben Konto an.
2. Lies das [GitHub-Teamhandbuch](output/pdf/STS2026_GitHub_Team_Handbook.pdf): Seite 5 erklärt die Annahme von Einladungen, Seiten 6–9 den Web-Workflow und Seite 19 eine Einstiegsübung vollständig im Browser.
3. Erstelle für jede Aufgabe einen Branch von `main`, committe deine Änderungen in diesen Branch und öffne einen PR mit dem Ziel `main`.
4. Bitte ein anderes Teammitglied um Review. Zu Beginn koordiniert die projektverantwortliche Person die Merges.

Ein Commit über die Website ist bereits auf GitHub gespeichert; ein zusätzlicher Push ist nicht nötig. Zum Ausführen von Code benötigst du die unter [Einrichtung](docs/SETUP.md) beschriebene Umgebung. Die ergänzende Dokumentation und die PDFs sind derzeit überwiegend auf Englisch.

## Aktueller Stand des Notebooks

[`vits-light_v1.ipynb`](vits-light_v1.ipynb) enthält Text- und Posterior-Encoder, ein Flow-Modell, Alignment- und Dauerkomponenten, einen Wellenform-Decoder, Verlustfunktionen und einen Trainer. Die **aktive Versuchskonfiguration** deckt weniger ab als die gesamte Architektur:

- LJSpeech mit einem fest eingetragenen Kaggle-Datenpfad und Audio mit 22.050 Hz.
- Aufruf von `overfit_train(total_steps=1500)` für einen einzigen festen Batch.
- Dauer- und KL-Verluste haben das Gewicht null; rekonstruktionsorientierte Verluste sind aktiv.
- Der GAN-Verlust ist weiterhin null; der Diskriminator ist ein Platzhalter.
- `evaluate(outputs, batch)` ist noch nicht implementiert.
- CUDA wird verwendet, wenn verfügbar, sonst die CPU; Apples MPS-Backend wird nicht ausgewählt.

Dies ist ein Ausgangspunkt für die Entwicklung, kein Nachweis einer vollständigen VITS-Implementierung, einer Evaluation mit Messergebnissen oder einer validierten Stimmkonvertierung. Das Notebook enthält keine gespeicherten Zellausgaben. Lies vor der Ausführung die [Einschränkungen der Baseline](docs/SETUP.md#current-limitations).

## Repository-Struktur

```text
STS2026/
  vits-light_v1.ipynb              Ursprüngliches Baseline-Notebook
  README.md                      Englische Startseite
  README.zh-CN.md                Chinesische Beschreibung
  README.de.md                   Deutsche Beschreibung
  README.fr.md                   Französische Beschreibung
  CONTRIBUTING.md                Regeln für Zusammenarbeit und Review
  requirements-notebook.txt       Notebook-Abhängigkeiten ohne PyTorch
  requirements-docs.txt           Abhängigkeiten für PDF-Erstellung und Prüfung
  .github/pull_request_template.md
  docs/
    SETUP.md                     Umgebung, Datensatz und Ausführung
    README.md                    Dokumentationsindex und Build-Anleitung
    build_github_handbook.py      Layout und Erstellung des Handbuchs
    handbook_web_content.py       Englische Inhalte zum Web-Workflow
    check_github_handbook.py      Prüfung von PDF-Struktur und Layout
  output/pdf/
    STS2026_GitHub_Team_Handbook.pdf
    ASR_VITS_Evaluation_Guide.pdf
```

`tmp/`, Daten, Umgebungen, Caches, Checkpoints und erzeugte Audiobeispiele bleiben lokal und werden ignoriert. Die beiden ausgewählten PDFs werden bewusst versioniert. Weitere Ausgabedateien erfordern eine Prüfung und eine Anpassung der Ignore-Regeln.

## Vereinbarungen im Team

- Ein kurzlebiger Branch pro Aufgabe; `main` enthält den geprüften Stand.
- Zuständigkeiten vor Notebook-Änderungen absprechen. Gleichzeitiges Ersetzen kompletter Notebooks per Web-Upload vermeiden.
- In jedem PR genau angeben, was geprüft wurde und was ungeprüft bleibt.
- Daten, Modellgewichte und Zugangsdaten nicht in Git aufnehmen; auch Notebook-Ausgaben prüfen.
- Branch-Schutz auf GitHub wird nicht vorausgesetzt. Die Teamregeln gelten auch, wenn die Oberfläche direkte Commits erlaubt.

Siehe [Beitragsrichtlinien](CONTRIBUTING.md) und [Dokumentation](docs/README.md).

## Validierung und Weiterverwendung

Bei der Repository-Vorbereitung werden nur Notebook-Struktur und Python-Syntax geprüft. Diese Prüfungen führen kein Training aus und belegen nicht die Korrektheit des Modells. Datenzugriff, kompatible PyTorch-/TorchAudio-Versionen, ein Evaluationsprotokoll mit zurückgehaltenen Daten und tatsächliche Versuchsergebnisse müssen noch festgelegt bzw. erarbeitet werden.

Es wurde noch keine Softwarelizenz gewählt. Vor Veröffentlichung oder Weiterverwendung sind Kursvorgaben, Codeherkunft sowie Bedingungen für Drittinhalte und Daten zu prüfen. Über eine Lizenz entscheidet das Team gemeinsam.
