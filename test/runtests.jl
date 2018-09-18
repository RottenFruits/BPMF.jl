using Test, Random, Statistics, BPMF

Random.seed!(0)

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

#################################################################
#       GBPMF
#################################################################
#init parameter
β₀ = 2
μ₀ = 0
D = 3
ν₀ = D
W₀ = one(zeros(D, D))
α = 2
T = 100
N = length(unique(R[:, 1]))
M = length(unique(R[:, 2]))

#learning
model = BPMF.GBPMFModel(D, α, β₀, μ₀, ν₀, W₀, N, M, T, R, [], [])
BPMF.fit(model)
BPMF.predict_all(model, 10)
BPMF.predict(model, R, 10)

println(sqrt(mean((R[:, 3] - BPMF.predict(model, R, 10)) .^ 2)))

#################################################################
#       VBPMF
#################################################################
#init parameter
N = length(unique(R[:, 1]))
M = length(unique(R[:, 2]))
D = 3
τ² = 1
L = 10

#learning
model = BPMF.VBPMFModel(R, N, M, D, [], [], τ², [], [], L)
BPMF.fit(model)

println(sqrt(mean((R[:, 3] - BPMF.predict(model, R)) .^ 2)))
