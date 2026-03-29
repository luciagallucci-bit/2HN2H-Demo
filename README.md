
================

# To Hit or Not To Hit 🎯

> An interactive Shiny demo for evaluating Knowledge Graph Embedding
> models through probabilistic ranking metrics.

[![R](https://img.shields.io/badge/R-%3E%3D4.1-blue)](https://www.r-project.org/)
[![Shiny](https://img.shields.io/badge/Shiny-1.7+-brightgreen)](https://shiny.posit.co/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

------------------------------------------------------------------------

## Authors

| Name                   | 
|------------------------|
| Gian Mario Sangiovanni |
| Lucia Gallucci         | 
| Lorenzo Balzotti       | 
| Donatella Firmani      | 
| Giovanna Jona Lasinio  | 
| Johannes Resin         | 

------------------------------------------------------------------------

## Overview

**To Hit or Not To Hit** is an interactive demo that explores the
evaluation of Knowledge Graph Embedding (KGE) models beyond standard
<Hits@k> metrics.

Rather than relying solely on binary ranking indicators, this tool
introduces **probabilistic Top-k Log Scores** derived from the model’s
full output distribution, alongside cross-fold consistency analysis and
tuple-level uncertainty quantification via **Chebyshev confidence
intervals**.

The demo is built around four KGE models evaluated on standard benchmark
datasets, with results aggregated across 15 cross-validation folds.

------------------------------------------------------------------------

## Features

- **Model Evaluation tab** — compare <Hits@k>, MRR, and Top-k Log Scores
  across folds and models with forest plots
- **Cross-Fold Analysis tab** — assess prediction consistency of
  individual triples across folds using a selectable reference fold
- **Triple Inspector tab** — drill into sampled triples with per-fold
  rank breakdowns, Chebyshev CIs, and Top-100 entity predictions
- Supports filtering triples by difficulty: Easy / Medium / Hard / High
  Variance / Random
- Downloadable results as CSV

------------------------------------------------------------------------

## Supported Datasets & Models

**Datasets**

- YAGO3-10
- WN18RR
- FB15k-237
- KINSHIPS

**Models**

- RotatE
- HolE
- MurE
- ComplEx

------------------------------------------------------------------------

## Installation

### Prerequisites

- R ≥ 4.1
- The following R packages:

``` r
install.packages(c(
  "shiny", "ggplot2", "dplyr", "tidyr", "purrr",
  "DT", "stringr", "bslib", "shinyWidgets",
  "arrow", "renv", "plotly", "shinycssloaders"
))
```

### Running the App

Clone the repository and launch the app from R:

``` r
# Clone the repo, then:
shiny::runApp("path/to/app")
```

Or directly from GitHub (requires `remotes`):

``` r
shiny::runGitHub("repo-name", "username")
```

> **Note:** The app expects pre-processed `.parquet` files (e.g.,
> `combined_YAGO3_10_RotatE.parquet`) and training files (e.g.,
> `train_YAGO3-10.txt`) in the working directory. Contact the authors
> for access to the data.

------------------------------------------------------------------------

## Methodology Notes

### Probabilistic Top-k Log Score

Standard <Hits@k> is a binary metric: a model either ranks the true tail
in the top k or it doesn’t. This demo introduces a **soft probabilistic
alternative**:

- If the true tail falls **within** top-k, its score is
  `-log(softmax_true)`
- If the true tail falls **outside** top-k, the score reflects the
  residual probability mass redistributed uniformly over the remaining
  entities

This formulation allows for a more nuanced comparison between models
with similar <Hits@k> but different confidence profiles.


------------------------------------------------------------------------

## Repository Structure

    .
    ├── app.R                          # Main Shiny application
    ├── combined_YAGO3_10_RotatE.parquet
    ├── combined_YAGO3_10_HolE.parquet
    ├── combined_YAGO3_10_MurE.parquet
    ├── combined_YAGO3_10_ComplEx.parquet
    ├── train_YAGO3-10.txt
    └── README.md

------------------------------------------------------------------------

## Citation

If you use this demo or the methodology in your work, please cite:

``` bibtex
@misc{2HN2H2026,
  title  = {To Hit or not to Hit: Computing accuracy of Knowledge Graph Completion Tasks with Proper Scores},
  author = {Gian Mario Sangiovanni, Lucia Gallucci, Lorenzo Balzotti, Donatella Firmani, Giovanna Jona Lasinio, Johannes Resin},
  year   = {2026}
}
```

------------------------------------------------------------------------

## License

This project is released under the [MIT License](LICENSE).
