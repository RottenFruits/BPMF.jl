# BPMF.jl

A Julia package for Bayesian Probabilistic Matrix Factorization (BPMF).

---
## How to install

You can install BPMF.jl running the following commands.

1. Into The Pkg REPL-mode

  - Enter the key`]` on the Julia REPL.

2. Install the package

  - Enter the following commond.

```julia
(v1.0) pkg> add https://github.com/RottenFruits/BPMF.jl
```


## Overview

This package is implementation bayesian probabilistic matrix factorization (BPMF).

Supported features:

- Gibbs sampling algorithm
- Variational Inference Algorithm

## Example

Here are examples.

At first read package and generate data.

```julia
using BPMF

#data
R = [
    0 0 7;
    0 1 6;
    0 2 7;
    0 3 4;
    0 4 5;
    0 5 4;
    1 0 6;
    1 1 7;
    1 3 4;
    1 4 3;
    1 5 4;
    2 1 3;
    2 2 3;
    2 3 1;
    2 4 1;
    3 0 1;
    3 1 2;
    3 2 2;
    3 3 3;
    3 4 3;
    3 5 4;
    4 0 1;
    4 2 1;
    4 3 2;
    4 4 3;
    4 5 3;
]
```

### Using Gibbs Sampling Algorithm

Following example is gibbs sampling algorithm.

```julia
N = length(unique(R[:, 1])) #number of unique users
M = length(unique(R[:, 2])) #number of unique items
D = 3 #number of latent dimensions
T = 100 #number of iterations
U = [] #user's latent factor
V = [] #item's latent factor
α = 2 #hyper parameter
β₀ = 2 #hyper parameter
μ₀ = 0 #hyper parameter
ν₀ = D #hyper parameter
W₀ = one(zeros(D, D)) #hyper parameter

#learning
gibbs_model = BPMF.GBPMFModel(R, N, M, D, T, U, V, α, β₀, μ₀, ν₀, W₀)
BPMF.fit(gibbs_model)

#predict new data
bi = 10 #burn-in
BPMF.predict(gibbs_model, R, bi)
```

### Using Variational Inference Algorithm

Following example is variational inference algorithm.

```julia
N = length(unique(R[:, 1])) #number of unique users
M = length(unique(R[:, 2])) #number of unique items
D = 3 #number of latent dimensions
L = 10 #number of iterations
U = [] #user's latent factor
V = [] #item's latent factor
τ² = 1 #hyper parameter
σ² = [] #hyper parameter
ρ² = [] #hyper parameter

#learning
variational_model = BPMF.VBPMFModel(R, N, M, D, L, U, V, τ², σ², ρ²)
BPMF.fit(variational_model)

#predict new data
BPMF.predict(variational_model, R)
```


## References


- Salakhutdinov, Ruslan, and Andriy Mnih. "Bayesian probabilistic matrix factorization using Markov chain Monte Carlo." Proceedings of the 25th international conference on Machine learning. ACM, 2008.

- Lim, Yew Jin, and Yee Whye Teh. "Variational Bayesian approach to movie rating prediction." Proceedings of KDD cup and workshop. Vol. 7. 2007.

- LIVESENSE Data Analytics Blog, BPMF(Bayesian Probabilistic Matrix Factorization)によるレコメンド, https://analytics.livesense.co.jp/entry/2017/12/05/105618
