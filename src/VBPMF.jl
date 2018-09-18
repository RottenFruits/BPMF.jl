# Variational Bayesian Probabilistic Matrix Factorization
#
# Reference
# ----------
#   Lim, Yew Jin, and Yee Whye Teh. "Variational Bayesian approach to movie rating prediction."
#   Proceedings of KDD cup and workshop. Vol. 7. 2007.
#
mutable struct VBPMFModel
    M::Array
    I::Int64
    J::Int64
    n::Int64
    U::Array
    V::Array
    τ²::Float64
    σ²::Array
    ρ²::Array
    L::Int64
end

function fit(model::VBPMFModel)
    M = model.M = model.M[sortperm(model.M[:, 1]), :]
    n = model.n
    I = model.I
    J = model.J
    K = size(M)[1]
    Ψ = zeros(n, n, J)
    model.σ² = ones(n)
    model.ρ² = ones(n)
    model.U = rand(I, n)
    model.V = rand(J, n)

    #1.nitialize Sj and tj for j = 1,...,J:
    S, t = initialize_S_t(model, n, J)

    for l = 1:model.L
        #2. Update Q(ui) for i = 1,...,I:
        model.U, S, t, τ²_tmp, σ²_tmp = update_U(model, Ψ, S, t, I)

        #3. Update Q(vj) for j = 1,...,J:
        model.V, Ψ, ρ²_tmp = update_V(model, Ψ, S, t, J)

        #update Learning the Variances
        model.σ², model.ρ², model.τ² = update_learning_variances(model, I, J, K, σ²_tmp, ρ²_tmp, τ²_tmp)
    end
end

function initialize_S_t(model::VBPMFModel, n, J)
    S = zeros(n, n, J)
    for j = 1:J
        S[:, :, j] = one(zeros(n, n))
    end
    t = zeros(J, n)
    return(S, t)
end

function update_U(model::VBPMFModel, Ψ, S, t, I)
    n = model.n
    M = model.M
    U = model.U
    V = model.V
    τ² = model.τ²
    σ² = model.σ²

    τ²_tmp = 0
    σ²_tmp = zeros(n)
    σ²_matrix = one(zeros(n, n)) .* (1 ./ σ²)

    for i = 1:I
        #(a) Compute Φi and ui:
        N_i = M[M[:, 1] .== i - 1, :] #user's rate item
        N_i[:, 1:2] .+= 1 #index start 1

        Φ = inv(σ²_matrix + sum((Ψ[:, :, N_i[:, 2]] .+ mean(V[N_i[:, 2], :], dims = 1)' * mean(V[N_i[:, 2], :], dims = 1)) / τ², dims = 3)[:, :, 1])
        u = Φ * sum((N_i[:, 3]' * V[N_i[:, 2], :]) / τ², dims = 1)'
        U[i, :] = u
        σ²_tmp  = σ²_tmp + diag(Φ)

        #(b) Update Sj and tj for j ∈ N(i), and discard Φi:
        idx = 1
        for j = N_i[:, 2]
            S[:, :, j] = S[:, :, j] + (Φ + U[i, :] * U[i, :]') / τ²
            t[j, :] = t[j, :] + ((N_i[idx, 3] * U[i, :]') / τ²)'
            τ²_tmp =  τ²_tmp + (N_i[idx, 3] ^ 2) - (2 * N_i[idx, 3] * (U[i, :]' * V[j, :])) + tr((Φ + U[i, :] * U[i, :]') * (Ψ[:, :, j] + V[j, :] * V[j, :]'))
            idx = idx + 1
        end
    end

    σ²_tmp = σ²_tmp + (mean(U, dims =1) .^ 2)'

    return(U, S, t, τ²_tmp, σ²_tmp)
end

function update_V(model::VBPMFModel, Ψ, S, t, J)
    V = zeros(J, model.n)
    ρ²_tmp = 0
    #3. Update Q(vj) for j = 1,...,J:
    for j = 1:J
        Ψ[:, :, j] = inv(S[:, :, j])
        V[j, :] = (Ψ[:, :, j] * t[j, :])'
        ρ²_tmp  = ρ²_tmp .+ diag(Ψ[:, :, j])
    end
    ρ²_tmp = ρ²_tmp + (mean(V, dims =1) .^ 2)'

    return(V, Ψ, ρ²_tmp)
end

function update_learning_variances(model::VBPMFModel, I, J, K, σ²_tmp, ρ²_tmp, τ²_tmp)
    σ² = (1/(I - 1)) * σ²_tmp
    ρ² = (1/(J- 1)) * ρ²_tmp
    τ² = (1/(K - 1)) * τ²_tmp
    return(σ², ρ² , τ²)
end

function predict(model::VBPMFModel, new_data)
    R_p = model.U * model.V'
    r = zeros(size(new_data)[1])
    for i in 1:size(new_data)[1]
        r[i] = R_p[new_data[i, 1] + 1, new_data[i, 2] + 1]
    end
    return r
end
