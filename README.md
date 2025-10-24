# Automatic-follow-up-for-incidental-radiology-findings
Develop a back end program to extract the positive identification of incidental findings from radiology plain text reports and ensure that patients have received a proper diagnosis in their EHR and receive appropriate follow up.

## Overview
This project uses NLP to automatically detect incidental findings from radiology reports
and verify appropriate follow-up actions in the EHR (FHIR-based). It aims to reduce missed
follow-ups that can harm patients.

## Components
- **NLP Extraction:** Detect incidental findings in plain-text reports.
- **EHR Checker:** Cross-reference EHR data via FHIR API.
- **Scheduler:** Automate periodic scans.
- **Alert System:** Notify clinicians/admin when follow-up is missing.

## Quick Start
```bash
git clone https://github.com/<your-username>/auto-followup.git
cd auto-followup
pip install -r requirements.txt
python src/main.py