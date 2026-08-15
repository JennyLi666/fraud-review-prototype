# Fraud Review Prototype — Streamlit Deployment

This repository contains the complete Streamlit showcase for the fraud-review capstone prototype.

All five interface areas remain visible: Overview, Review queue, Data uploads, Monitoring, and Model updates. The included synthetic upload file can be found at `prototype_phase12/examples/demo_claim_upload.parquet`.

## Deploy on Streamlit Community Cloud

1. Push the contents of this folder to the root of a GitHub repository.
2. Open <https://share.streamlit.io/> and choose **Create app**.
3. Select the repository and the `main` branch.
4. Set **Main file path** to `prototype_phase12/app.py`.
5. In advanced settings, select Python 3.12.
6. Deploy and test the generated `streamlit.app` URL on a phone.

No API key or secret is required.

## Local check

From the repository root:

```powershell
python -m pip install -r prototype_phase12/requirements.txt
python -m streamlit run prototype_phase12/app.py
```

## Important

- Only demonstration and de-identified artifacts are included.
- Do not add confidential claim data, training datasets, local SQLite files, logs, or secrets to this repository.
- Hosted local state is temporary and is not a production database.
- The model-update screen is included for presentation. Running a new retraining job requires the private historical train/test datasets, which are intentionally excluded from this public package.
