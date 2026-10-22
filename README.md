# Automated Quality Control in Manufacturing

A project that uses image analysis to detect defects on mobile phone screens.

## What's inside

- frontend/ - the project website
- Mobile.h5 - the trained MobileNet model
- mobile-notebook.ipynb - the notebook used to train the model
- screen.py - the Streamlit app that classifies uploaded images
- requirements.txt - the Python packages needed

## How to run the classifier

Install the packages with: pip install -r requirements.txt
Then start the app with: streamlit run screen.py
Open http://localhost:8501 in your browser and upload an image.
