# Cardiomyocyte Ageing Analysis App

A Streamlit app for analysing cardiomyocyte microscopy images and videos for signs of ageing. It extracts morphological, structural and motion features from each file, combines them into an ageing score and compares files across culture days.

## Features

- **Input:** images, videos and Nikon `.nd2` microscopy files. The culture day is read from the file name.
- **Feature extraction:**
  - Cell morphology
  - Sarcomere organisation
  - Mitochondrial features
  - Nuclear features
  - Contraction motion from video frames
- **Ageing score:** the features are combined into an advanced ageing score with the contributing factors listed.
- **Multi-file analysis:** PCA, UMAP and k-means views across files, with Isolation Forest outlier detection.
- **Performance:** frame sampling, caching and parallel processing for large videos.

> This is a prototype. The ageing score is a rule-based combination of image features and has not yet been validated against biological ageing markers.

## Running locally

```bash
pip install streamlit numpy pandas opencv-python plotly scikit-learn umap-learn nd2
streamlit run app.py
```

## Related repositories

- [Cardiomyocyte-Ageing-Nucleus-Data-Analysis](https://github.com/datascintist-abusufian/Cardiomyocyte-Ageing-Nucleus-Data-Analysis): nuclear ageing biomarkers
- [Cardiomyocyte-cell-motion-analysis](https://github.com/datascintist-abusufian/Cardiomyocyte-cell-motion-analysis): cardiomyocyte morphodynamic simulator

## Licence

No licence has been declared yet.

## Author

Md Abu Sufian - ORCID [0009-0007-3503-6942](https://orcid.org/0009-0007-3503-6942)
