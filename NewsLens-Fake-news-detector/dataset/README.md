# Dataset instructions

This project expects two CSV files:

- `Fake.csv` — articles labelled as fake (label 0)
- `True.csv` — articles labelled as real (label 1)

Each CSV must contain a `text` column. A `title` column is recommended but optional.

## Suggested dataset

The reference project uses the ISOT Fake News Dataset. Obtain the dataset from its original/authorized distribution page or a reputable dataset host. Review the dataset's terms and citation requirements before use.

Place the files in this folder using the exact names above. The raw dataset is intentionally not bundled in this repository because dataset redistribution rights and repository size limits should be checked first.

The labels represent the dataset's annotations; they are not an independent verification of every claim in an article.
