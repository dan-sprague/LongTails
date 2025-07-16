# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LongTails is a Julia package for statistical analysis of transcriptomic data using negative binomial regression with dispersion trend modeling. The package focuses on modeling overdispersion in count data and implements Cox-Reid regularization for parameter estimation.

## Key Dependencies

- **Core Julia packages**: Distributions, Optim, SpecialFunctions, LinearAlgebra, StatsBase
- **Data handling**: DataFrames, CSV
- **Visualization**: Plots, StatsPlots
- **Statistical modeling**: GLM

## Architecture

The codebase is structured around several key statistical concepts:

### Core Components

1. **Negative Binomial Modeling** (`src/negbin2.jl`):
   - `NegBin2` struct for parameterized negative binomial distributions
   - Cox-Reid regularized likelihood functions
   - Method of moments estimation

2. **Transcriptome Simulation** (`src/transcriptome.jl`):
   - `PowerLaw` distribution for modeling gene expression
   - `Transcriptome` struct for simulating RNA-seq data
   - Fold change simulation with `FC!` function

3. **Dispersion Trend Modeling** (`src/dispersion_trend.jl`):
   - `fit_dispersion_trend` function for iterative parameter estimation
   - `atr` function implementing the dispersion-mean relationship: α = a₁/μ + a₀

4. **Likelihood Functions** (`src/likelihood.jl`):
   - Various negative log-likelihood functions with Cox-Reid regularization
   - Gamma regression for dispersion parameter modeling

5. **Utilities** (`src/utils.jl`, `src/math.jl`):
   - Data cleaning and normalization functions
   - Mathematical helper functions

## Development Workflow

The primary development occurs in `dev.jl`, which contains:
- Experimental code and parameter fitting
- Visualization and plotting commands
- Integration testing of different components

## Key Mathematical Concepts

- **Dispersion-mean relationship**: α(μ) = a₁/μ + a₀
- **Cox-Reid regularization**: Adds penalty term 0.5 * log(det(X'WX)) to likelihood
- **Method of moments**: Initial parameter estimation using sample variance and mean

## Running the Code

Execute `julia dev.jl` to run the main development script, which:
1. Simulates transcriptomic data
2. Fits negative binomial models with dispersion trends
3. Generates diagnostic plots

## File Structure

- `src/LongTails.jl`: Main module (currently minimal)
- `src/negbin2.jl`: Negative binomial distribution implementation
- `src/transcriptome.jl`: Transcriptome simulation framework
- `src/dispersion_trend.jl`: Dispersion trend fitting algorithms
- `src/likelihood.jl`: Likelihood function implementations
- `src/utils.jl`, `src/math.jl`: Utility functions
- `dev.jl`: Development and testing script