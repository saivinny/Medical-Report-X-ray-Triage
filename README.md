# TriageNet — Medical Report + X-ray Triage Assistant

**GitHub:** [View Repository](https://github.com/saivinny/Medical-Report-X-ray-Triage)

**Module:** 55-710603 — Advanced Artificial Intelligence Projects in Data Science

**Team:** Sai Vinny Reddy Garlapati (35064131) · Sai Sijju Reddy Garlapati (35064180) ·
Kalkiram Reddy Guduru (35062243) · Shruthi Manupuri (35062249) · Makarand Mangunoori (35062270)

---

## 1. What this is

TriageNet is a multimodal AI prototype that reads a **chest X-ray** and the accompanying
**clinical note** together, and classifies the case into one of three triage bands:

- `Normal`
- `Needs Attention`
- `Urgent`

It combines a CNN image branch (DenseNet-121) with an NLP text branch (ClinicalBERT / BioBERT),
fused with a late-fusion MLP head, following the architecture in our project proposal.

It is a **decision-support prototype for coursework purposes** — not a certified medical device,
and not intended to replace a clinician.

## 2. Repository structure

```
triagenet-project/
├── README.md                  <- you are here
├── requirements.txt
├── data/
│   ├── raw/                   <- put real datasets here (see "Datasets" below)
│   └── sample/                <- small synthetic dataset so the pipeline runs out of the box
├── src/
│   ├── config.py              <- shared constants (labels, paths, hyperparameters)
│   ├── data_preprocessing.py  <- Dataset classes + image/text preprocessing
│   ├── image_model.py         <- DenseNet-121 CNN branch
│   ├── text_model.py          <- ClinicalBERT / BioBERT text branch
│   ├── fusion_model.py        <- late-fusion multimodal classifier
│   ├── train.py                <- CLI training script (image / text / fusion)
│   ├── evaluate.py            <- metrics: recall, F1, AUC-ROC, specificity, confusion matrix
│   └── explainability.py      <- Grad-CAM (image) + token saliency (text)
├── webapp/
│   └── triage-assistant.html  <- in-browser interactive demo (see webapp/README.md)
└── tests/
    └── test_pipeline.py       <- sanity tests for model shapes / forward passes
```

## 3. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. A GPU is recommended for training (Google Colab is fine) but everything
runs on CPU with the small synthetic sample data for testing purposes.

## 4. Datasets

**Included:** `data/sample/` contains a small **synthetic** dataset (fabricated clinical notes and
procedurally generated placeholder X-ray-shaped images) so every script can be run and tested
immediately, end to end, without waiting on data access approvals.

**For real results**, swap in credentialed medical datasets, e.g.:

- Images: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) or
  [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) (PhysioNet credentialing required)
- Notes: [MIMIC-III / MIMIC-IV clinical notes](https://physionet.org/content/mimiciii/) (PhysioNet
  credentialing + CITI training required), or a de-identified note set of your own.

Once approved, drop images under `data/raw/xrays/<label>/` and notes into
`data/raw/clinical_notes.csv` (columns: `note_id,note_text,image_path,label`), then point
`--data-dir` at `data/raw` instead of `data/sample`.

## 5. Running it

```bash
# Train each branch separately, then the fused model
python src/train.py --modality image  --data-dir data/sample --epochs 10
python src/train.py --modality text   --data-dir data/sample --epochs 10
python src/train.py --modality fusion --data-dir data/sample --epochs 10

# Evaluate and compare all three against the held-out split
python src/evaluate.py --data-dir data/sample --models-dir models

# Explainability on a single example
python src/explainability.py --image data/sample/xrays/urgent/sample_001.png \
                              --note "severe chest pain radiating to left arm"
```

## 6. Web demo

`webapp/triage-assistant.html` is a standalone, browser-only interactive mock of the same
pipeline (no install needed) — useful for the live demo in Week 11/12. It uses lightweight
JS heuristics to *simulate* the trained models' behaviour for presentation purposes; the real
models live in `src/`.

## 7. Evaluation

The models are evaluated using **Urgent Recall, Macro F1-score, AUC-ROC, and Specificity**. Urgent Recall is the primary metric, with a target of **≥90%**.

| Model  | Urgent Recall | Macro F1 | AUC-ROC | Specificity |
|--------|---------------|----------|---------|-------------|
| Image  | 0.000         | 0.167    | 1.000   | 1.000       |
| Text   | 1.000         | 1.000    | 1.000   | 1.000       |
| Fusion | 0.000         | 0.222    | 0.500   | 0.500       |

> **Note:** Evaluation uses only **3 test samples**, so results are for prototype demonstration and not clinical validation.

## 8. Team & GitHub Workflow

GitHub is used for **version control and team collaboration**, covering data processing, model development, evaluation, explainability, and the web application.