# Automatic-follow-up-for-incidental-radiology-findings
Develop a back end program to extract the positive identification of incidental findings from radiology plain text reports and ensure that patients have received a proper diagnosis in their EHR and receive appropriate follow up.

## Overview
This project uses NLP to automatically detect incidental findings from radiology reports
and verify appropriate follow-up actions in the EHR (FHIR-based). It aims to reduce missed
follow-ups that can harm patients.

## Components
- **NLP Extraction:** Detect incidental findings in plain-text reports.
- **EHR Checker:** Cross-reference EHR data via FHIR API.
- **Alert System:** Notify clinicians/admin when follow-up is missing.

## Path
```
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── 500_labeled_nodules.csv     # raw radiology reports (de-identified)
│   ├── findings_nodule.csv         # parsed data
│   └── predictions_output.csv      # small sample reports for demo
│
├── src/
│   ├── Findings.py
│   ├── nlp_pipeline.py             # NLP Model trained with bioBert library, first version
│   ├── predict.py                  # Input raw data to predict pulmonary nodules present/absent
│   └── Model.py                    # NLP Model trained with bioBert library, second version
```


## Quick Start

1.  Run the model first:
```bash
python src/nlp_pipeline.py
```
2. A new folder "models" will be generated, then run the code for prediction:

```bash
python src/predict.py
```