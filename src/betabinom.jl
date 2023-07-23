using SpecialFunctions
using Distributions
using Optim
using LinearAlgebra

mutable struct BetaBinomModel{T<:Integer}

    # Number of successes per observation
    success::Vector{T}

    # Total number of trials per observation
    total::Vector{T}

    # An indicator that the MLE optimization converged
    converged::Bool

    # The log of the MLE: log(alpha), log(beta)
    params::Vector

    # The variance/covariance matrix of the MLE (for log-scale parameters)
    vcov::Matrix
end

function BetaBinomModel(success, total)
    return BetaBinomModel(success, total, false, zeros(2), zeros(2, 2))
end

function loglike(
    bb::BetaBinomModel,
    lalpha::Float64,
    lbeta::Float64,
    lalpha0::Float64,
    lbeta0::Float64,
    lam::Float64,
)

    (; success, total, params) = bb

    alpha, beta = exp(lalpha), exp(lbeta)
    alpha0, beta0 = exp(lalpha0), exp(lbeta0)

    ll = 0.0
    for i in eachindex(success)
        n, x = total[i], success[i]
        ll += logabsgamma(x + alpha)[1] + logabsgamma(n - x + beta)[1] + logabsgamma(alpha + beta)[1]
        ll -= logabsgamma(n + alpha + beta)[1] + logabsgamma(alpha)[1] + logabsgamma(beta)[1]
    end

    n = length(success)
    ll -= n * lam * (lalpha - lalpha0)^2
    ll -= n * lam * (lbeta - lbeta0)^2

    return ll
end

function score!(
    bb::BetaBinomModel,
    G::Vector{Float64},
    lalpha::Float64,
    lbeta::Float64,
    lalpha0::Float64,
    lbeta0::Float64,
    lam::Float64,
)

    (; success, total) = bb

    alpha, beta = exp(lalpha), exp(lbeta)
    alpha0, beta0 = exp(lalpha0), exp(lbeta0)

    ga, gb = 0.0, 0.0
    for i in eachindex(success)
        n, x = total[i], success[i]
        dab = digamma(alpha + beta)
        dnab = digamma(n + alpha + beta)
        ga += digamma(x + alpha) + dab - dnab - digamma(alpha)
        gb += digamma(n - x + beta) + dab - dnab - digamma(beta)
    end

    # Adjust for the change of variables
    G .= [ga * alpha, gb * beta]

    n = length(total)
    G[1] -= 2 * n * lam * (lalpha - lalpha0)
    G[2] -= 2 * n * lam * (lbeta - lbeta0)
end

function hess!(
    bb::BetaBinomModel,
    G::Vector,
    H::Matrix,
    lalpha::Float64,
    lbeta::Float64,
    lalpha0::Float64,
    lbeta0::Float64,
    lam::Float64,
)

    (; success, total) = bb

    alpha, beta = exp(lalpha), exp(lbeta)
    alpha0, beta0 = exp(lalpha0), exp(lbeta0)

    # Get the unpenalized score with respect to non-logged parameters
    score!(bb, G, lalpha, lbeta, lalpha0, lbeta0, 0.0)
    G[1] /= alpha
    G[2] /= beta

    # Get the Hessian of the log-likelihood with respect to the non-logged parameters
    H .= 0
    for i in eachindex(success)
        n, x = total[i], success[i]
        tab = trigamma(alpha + beta)
        tnab = trigamma(n + alpha + beta)
        H[1, 1] += trigamma(x + alpha) + tab - tnab - trigamma(alpha)
        H[2, 2] += trigamma(n - x + beta) + tab - tnab - trigamma(beta)
        H[1, 2] += tab - tnab
    end

    # Adjust for the change of variables, this is the unpenallized Hessian with
    # respect to the logged variables.
    H[1, 1] = alpha * (alpha * H[1, 1] + G[1])
    H[2, 2] = beta * (beta * H[2, 2] + G[2])
    H[1, 2] = alpha * beta * H[1, 2]

    H[2, 1] = H[1, 2]

    # Account for the penalty
    n = length(total)
    H[1, 1] -= 2 * n * lam
    H[2, 2] -= 2 * n * lam

    # Convert the gradient back to log-scale variables
    G[1] *= alpha
    G[2] *= beta
end

function fit(::Type{BetaBinomModel}, success, total; lalpha0=0.0, lbeta0=0.0, lam=0.0)
    bb = BetaBinomModel(success, total)
    fit!(bb, lalpha0, lbeta0, lam)
    return bb
end

function fit!(bb::BetaBinomModel, lalpha0::Float64, lbeta0::Float64, lam::Float64)

    (; success, total) = bb

    f = x -> -loglike(bb, x[1], x[2], lalpha0, lbeta0, lam)

    # Gradient-free optimization
    x0 = [1.0, 1.0]
    r = optimize(f, x0, NelderMead())#, Optim.Options(iterations=20))
    x1 = Optim.minimizer(r)

    # First order optimization
    g! = function (G, x)
        score!(bb, G, x[1], x[2], lalpha0, lbeta0, lam)
        G .*= -1
    end
    r = optimize(f, g!, x1, LBFGS())
    bb.converged = Optim.converged(r)
    bb.params = Optim.minimizer(r)

    G = zeros(2)
    H = zeros(2, 2)
    lalpha, lbeta = bb.params
    hess!(bb, G, H, lalpha, lbeta, lalpha0, lbeta0, lam)
    bb.vcov = -pinv(H)
end

function coef(bb::BetaBinomModel)
    return bb.params
end

function vcov(bb::BetaBinomModel)
    return bb.vcov
end
