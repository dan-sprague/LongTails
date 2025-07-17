# LongTails.jl

![Long Tailed Lizard](./takydromus_sexlineatus.jpg)


This is a scratch Julia implementation of the DESeq2 algorithm for differential expression analysis from RNAseq datasets. More generally, DESeq2 provides robust estimates of dispersion for Negative Binomial GLMs when there is a dispersion dependence on mean observed counts. This enables NB GLMs to be fit to noisy datasets where few replicates are available.


This repository is a work in progress


## Implementation Goals


1. Significant speedup on large datasets using Julia's native compilation to LLVM and threading capabilities.
2. Direct from command line execution, including design formula.
3. Translate tximport and apeglm to Julia for start-to-finish analysis
4. Minimize dependencies
5. Hard coded gradients


## Currently finished

1. Simulation of transcriptome count datasets following an arbitrary regression formula for testing.
    - PowerLaw distribution imeplementation
    - Returns count matrix, sampled coefficients, and other parameters
    - Essentially simulates the generative process DESeq2 is estimating with its model
2. Native NegativeBinomial2 implementation
    - NB distribution formulated in terms of mean and dispersion
    - sampling done via gamma-poisson distribution
3. Normalization and scaling factors
    - Implemented effective length normalization and sequencing depth normalization