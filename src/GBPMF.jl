# Gibbs Sampling Bayesian Probabilistic Matrix Factorization
#
# Reference
# ----------
#   Salakhutdinov, Ruslan, and Andriy Mnih. "Bayesian probabilistic matrix factorization using Markov chain Monte Carlo."
#   Proceedings of the 25th international conference on Machine learning. ACM, 2008.
#
mutable struct GBPMFModel
    R::Array
    N::Int64
    M::Int64
    D::Int64
    T::Int64
    U::Array
    V::Array
    α::Float64
    β₀::Float64
    μ₀::Float64
    ν₀::Float64
    W₀::Array
end

function fit(model::GBPMFModel)
    #set initial value
    model.U = zeros((model.T, model.N, model.D)) #user latent matrix
    model.V = zeros((model.T, model.M, model.D)) #item latent matrix
    model.U[1, :, :] = rand(model.N, model.D)
    model.V[1, :, :] = rand(model.M, model.D)

    T = model.T
    N = model.N
    M = model.M

    for t in 1:(T-1) #1 is initial value
        #u matrix
        Θᵤ = hyper_prameter_posterior(model, model.U[t, :, :])
        μᵤ, Λᵤ = hyper_parameter_sampling(Θᵤ[1], Θᵤ[2], Θᵤ[3], Θᵤ[4])

        #v matrix
        Θᵥ = hyper_prameter_posterior(model, model.V[t, :, :])
        μᵥ, Λᵥ = hyper_parameter_sampling(Θᵥ[1], Θᵥ[2], Θᵥ[3], Θᵥ[4])

        for n in 1:N
            μᵢ_asterisk, Λᵢ_asterisk_inv = u_matrix_posterior(model, model.V[t, :, :], μᵤ, Λᵤ, n)
            model.U[t + 1, n, :] = latent_matrix_sampling(μᵢ_asterisk, Λᵢ_asterisk_inv)
        end

        for m in 1:M
            μᵢ_asterisk, Λᵢ_asterisk_inv = v_matrix_posterior(model, model.U[t + 1, :, :], μᵥ, Λᵥ, m)
            model.V[t + 1, m, :] = latent_matrix_sampling(μᵢ_asterisk, Λᵢ_asterisk_inv)
        end

    end
end

function hyper_prameter_posterior(model::GBPMFModel, latent_matrix)
    N = size(latent_matrix)[2]
    β₀ = model.β₀
    μ₀ = model.μ₀
    ν₀ = model.ν₀
    W₀ = model.W₀

    latent_matrix_bar = (1 / N) * sum(latent_matrix[:, :], dims = 1)
    S̄ = (1 / N) .* sum([latent_matrix[i, :]' .* latent_matrix[i, :] for i = 1:N], dims = 1)[1]

    μ₀_asterisk = ((β₀ * μ₀ .+ N * latent_matrix_bar) / (β₀ + N))[1, :]
    β₀_asterisk = β₀ + N
    ν₀_asterisk = ν₀ + N

    W₀_asterisk = inv(inv(W₀) .+ N .* S̄ .+ ((β₀ * N) / (β₀ + N)) .* (μ₀ .- latent_matrix_bar)' * (μ₀ .- latent_matrix_bar))
    W₀_asterisk = Array(Symmetric(W₀_asterisk))

    return μ₀_asterisk, β₀_asterisk, ν₀_asterisk, W₀_asterisk
end

function hyper_parameter_sampling(μ₀_asterisk, β₀_asterisk, ν₀_asterisk, W₀_asterisk)
    Λ = rand(Wishart(ν₀_asterisk, W₀_asterisk), 1)[1]
    μ = rand(MvNormal(μ₀_asterisk, Array(Symmetric(inv(β₀_asterisk * Λ)))), 1)

    return μ, Λ
end

function u_matrix_posterior(model::GBPMFModel, latent_matrix, μᵤ, Λᵤ, n)
    R = model.R
    α = model.α

    r = R[R[:, 1] .==  n - 1, :] #user's rate item
    nonzero_ix = (R[R[:, 1] .==  n - 1, :][:, 2] .+ 1) #user's rate item index

    if nonzero_ix == [] #when no rate then return random value
        μᵢ_asterisk = zeros(size(latent_matrix)[2])
        Λᵢ_asterisk = one(zeros(size(latent_matrix)[2], size(latent_matrix)[2]))
        return μᵢ_asterisk, Λᵢ_asterisk
    end

    Λᵢ_asterisk = Λᵤ .+ α * sum([latent_matrix[i, :]' .*  latent_matrix[i, :] for i in nonzero_ix], dims = 1)[1]
    Λᵢ_asterisk_inv = Array(Symmetric(inv(Λᵢ_asterisk)))

    μᵢ_asterisk = (inv(Λᵢ_asterisk) * ((α * sum([latent_matrix[i, :] .* r[r[:, 2] .== i - 1, :][:, 3] for i in nonzero_ix], dims = 1)[1]') + (Λᵤ * μᵤ)')')[:, 1]

    return μᵢ_asterisk, Λᵢ_asterisk_inv
end

function v_matrix_posterior(model::GBPMFModel, latent_matrix, μᵥ, Λᵥ, m)
    R = model.R
    α = model.α

    r = R[R[:, 2] .==  m - 1, :] #item rate user
    nonzero_ix = (R[R[:, 2] .==  m - 1, :][:, 1] .+ 1) #item rate user index

    if nonzero_ix == [] #when no rate then return random value
        μᵢ_asterisk = zeros(size(latent_matrix)[2])
        Λᵢ_asterisk = one(zeros(size(latent_matrix)[2], size(latent_matrix)[2]))
        return μᵢ_asterisk, Λᵢ_asterisk
    end

    Λᵢ_asterisk = Λᵥ .+ α * sum([latent_matrix[i, :]' .*  latent_matrix[i, :] for i in nonzero_ix], dims = 1)[1]
    Λᵢ_asterisk_inv = Array(Symmetric(inv(Λᵢ_asterisk)))

    μᵢ_asterisk = (inv(Λᵢ_asterisk) * ((α * sum([latent_matrix[i, :] .* r[r[:, 1] .== i - 1, :][:, 3] for i in nonzero_ix], dims = 1)[1]') + (Λᵥ * μᵥ)')')[:, 1]

    return μᵢ_asterisk, Λᵢ_asterisk_inv
end

function latent_matrix_sampling(μᵢ_asterisk, Λᵢ_asterisk_inv)
    return  rand(MvNormal(μᵢ_asterisk, Λᵢ_asterisk_inv), 1)
end


function predict_all(model::GBPMFModel, burn_in)
    T = model.T
    sum([model.U[i, :, :] *  model.V[i,:,  :]' for i = burn_in:T], dims = 1)[1]  / (T - burn_in )
end

function predict(model::GBPMFModel, new_data, burn_in)
    R_p = predict_all(model, burn_in)
    r = zeros(size(new_data)[1])
    for i in 1:size(new_data)[1]
        r[i] = R_p[new_data[i, 1] + 1, new_data[i, 2] + 1]
    end
    return r
end
