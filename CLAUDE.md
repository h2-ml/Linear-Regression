# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a linear regression learning repository containing Jupyter notebooks and Python scripts that demonstrate:
- Linear regression concepts with R-squared calculations
- Gradient descent algorithms (batch, mini-batch, stochastic)
- Correlation analysis and Pearson's correlation
- Machine learning implementations using scikit-learn

## Key Libraries and Dependencies
- **Polars**: Primary data manipulation (used instead of pandas in notebooks)
- **plotnine**: Grammar of graphics plotting (ggplot2-style)
- **scikit-learn**: Machine learning algorithms and datasets
- **numpy**: Numerical computations
- **matplotlib**: Traditional plotting
- **pandas**: Alternative data manipulation (some notebooks)

## Development Commands
```bash
# Run Python scripts
python gradient-descent-claude.py
python hello.py

# Launch Jupyter notebooks
jupyter notebook
```

## Code Architecture
The repository contains two main types of implementations:

### Notebooks (`*.ipynb`)
- `1.2.R-Squared-Example.ipynb`: R-squared calculations with polars/plotnine
- `Gradient-Descent.ipynb`: Theoretical gradient descent implementation
- `Gradient-Descent-claude.ipynb`: Enhanced gradient descent with visualization
- `2.1.What-is-Correlation.ipynb`: Correlation analysis
- `Pearsons.ipynb`: Pearson correlation coefficient implementations

### Python Scripts
- `gradient-descent-claude.py`: Complete gradient descent implementation with multiple variants (batch, mini-batch, stochastic), cost function visualization, and scikit-learn comparisons

## Data Visualization Patterns
- Uses plotnine with ggplot2-style grammar of graphics
- Data often converted to long-form using `unpivot()` for proper color mapping
- Generates PNG output files for visualizations (numbered 1-7)
- Matplotlib for traditional scientific plotting

## Key Functions and Utilities
- `r_squared()`: Calculate R-squared metrics with detailed output
- `plot_yY()`: Overlay actual vs predicted values visualization
- `compute_cost()`: Mean squared error cost function
- `gradient_descent()`: Full gradient descent implementation with history tracking
- `stochastic_gradient_descent()`: SGD variant with configurable batch sizes

## Development Notes
- Python 3.12+ required (per pyproject.toml)
- Uses both traditional matplotlib and modern plotnine for different visualization needs
- Emphasizes educational implementations from scratch alongside scikit-learn comparisons
- Generates multiple visualization files to demonstrate algorithm convergence and behavior