using Lasso

mutable struct BetaBinomSeq{T<:Integer}

    # Number of successes per observation
    success::Vector{T}

    # Total number of trials per observation
    total::Vector{T}

    # An indicator that the MLE optimization converged
    converged::Bool

    # The penalty parameter for the fused lasso
    fuse_wt

    # Shape parameters on the log scale
    lalpha::Vector{Float64}
    lbeta::Vector{Float64}

    mean::Vector{Float64}
    icc::Vector{Float64}
end

function BetaBinomSeq(success::AbstractVector, total::AbstractVector; fuse_wt=0.0)
    @assert length(success) == length(total)
    n = length(success)
    lalpha = zeros(n)
    lbeta = zeros(n)
    mean = zeros(n)
    icc = zeros(n)

    return BetaBinomSeq(success, total, false, fuse_wt, lalpha, lbeta, mean, icc)
end

function loglike_single(
    success::T,
    total::T,
    lalpha0::Float64,
    lbeta0::Float64,
    par::Vector{Float64},
    lam::Float64,
) where {T<:Integer}
    lalpha, lbeta = par[1], par[2]
    alpha, beta = exp(lalpha), exp(lbeta)
    alpha0, beta0 = exp(lalpha0), exp(lbeta0)

    ll = 0.0
    ll += logabsgamma(success + alpha)[1] + logabsgamma(total - success + beta)[1] +
            logabsgamma(alpha + beta)[1]
    ll -= logabsgamma(total + alpha + beta)[1] + logabsgamma(alpha)[1] + logabsgamma(beta)[1]

    ll -= lam * (lalpha - lalpha0)^2
    ll -= lam * (lbeta - lbeta0)^2

    return ll
end

function score_single!(
    success::T,
    total::T,
    lalpha0::Float64,
    lbeta0::Float64,
    par::Vector{Float64},
    G::Vector{Float64},
    lam::Float64,
) where {T<:Integer}

    lalpha, lbeta = par[1], par[2]
    alpha, beta = exp(lalpha), exp(lbeta)
    alpha0, beta0 = exp(lalpha0), exp(lbeta0)

    ga, gb = 0.0, 0.0
    dab = digamma(alpha + beta)
    dnab = digamma(total + alpha + beta)
    ga += digamma(success + alpha) + dab - dnab - digamma(alpha)
    gb += digamma(total - success + beta) + dab - dnab - digamma(beta)

    # Adjust for the change of variables
    G .= [ga * alpha, gb * beta]

    G[1] -= 2 * lam * (lalpha - lalpha0)
    G[2] -= 2 * lam * (lbeta - lbeta0)
end

function admm!(success::Int32, total::Int32, lalpha::Float64, lbeta::Float64,
               par::Vector{Float64}, par1::Vector{Float64}, G::Vector{Float64},
               rho::Float64, step0::Float64)

    cnvrg = false
    gnrm = 0.0
    for k in 1:20
        cnvrg, gnrm = admm_step!(success, total, lalpha, lbeta, par, par1, G, rho, step0)
    end

    return cnvrg, gnrm
end

function admm_step!(success::Int32, total::Int32, lalpha0::Float64, lbeta0::Float64,
                    par::Vector{Float64}, par1::Vector{Float64}, G::Vector{Float64},
                    rho::Float64, step0::Float64)

    f0 = loglike_single(success, total, lalpha0, lbeta0, par, rho)
    score_single!(success, total, lalpha0, lbeta0, par, G, rho)

    while step0 >= 1e-20
        par1 .= par + step0*G
        f1 = loglike_single(success, total, lalpha0, lbeta0, par1, rho)
        if f1 > f0
            par .= par1
            return true, norm(G)
        end
        step0 /= 2
    end
    return false, norm(G)
end

function gradstep_fuse(bb::BetaBinomSeq; step0::Float64=1.0, rho::Float64=1.0)

    (; success, total, lalpha, lbeta) = bb

    n = length(lalpha)
    G = zeros(2)
    lam = 1.0
    par = zeros(2)
    par1 = zeros(2)
    nfail = 0
    maxg, avgg = 0.0, 0.0

    for i in 1:n

        par[1] = lalpha[i]
        par[2] = lbeta[i]
        cnvrg, gnrm = admm!(success[i], total[i], lalpha[i], lbeta[i], par, par1, G,
                            rho, step0)
        if cnvrg
            lalpha[i], lbeta[i] = par[1], par[2]
        else
            nfail += 1
        end
        avgg += gnrm
        maxg = max(maxg, gnrm)
    end

    avgg /= n

    return nfail, maxg, avgg
end

function set_start!(bb::BetaBinomSeq)

    (; success, total, lalpha, lbeta) = bb

    icc = 0.5

    for i in eachindex(total)
        # Method of moments
        n = total[i]
        p = (success[i] + 1) / (n + 2)
        v = n*p*(1-p)*(1 + (n-1)*icc)
        m1 = n*p
        m2 = m1^2 + v
        den = n*(m2/m1 - m1 - 1) + m1
        alpha = (n*m1 - m2) / den
        beta = (n - m1)*(n - m2/m1) / den

        lalpha[i] = log(alpha)
        lbeta[i] = log(beta)
    end
end

function get_mean_icc!(bb::BetaBinomSeq)
    (; lalpha, lbeta, total) = bb
    a = exp.(bb.lalpha)
    b = exp.(bb.lbeta)

    bb.mean .= a ./ (a + b)
    bb.icc .= 1 ./ (a + b .+ 1)
end

function reset_shape!(bb::BetaBinomSeq)

    (; lalpha, lbeta, mean, icc, total) = bb

    # Beta-binomial variance
    va = total .* mean .* (1 .- mean)
    va .*= 1 .+ (total .- 1) .* icc

    # Raw (uncentered) first and second beta-binomial moments
    m1 = total .* mean
    m2 = m1.^2 + va
    r = m2 ./ m1

    den = total.*(r - m1 .- 1) + m1
    lalpha .= total.*m1 - m2
    lalpha ./= den
    lalpha .= log.(lalpha)

    lbeta .= (total .- m1) .* (total - r)
    lbeta ./= den
    lbeta .= log.(lbeta)
end

function fuse!(bb::BetaBinomSeq)

    (; mean, icc, fuse_wt) = bb

    get_mean_icc!(bb)

    r = fit(FusedLasso, icc, fuse_wt)
    icc .= coef(r)

    d = mean .- 0.5
    s = sign.(d)
    d .= abs.(d)
    r = fit(FusedLasso, d, fuse_wt)
    mean .= 0.5 .+ s.*coef(r)

    reset_shape!(bb)
end

function fit!(bb::BetaBinomSeq; start=nothing, maxiter::Int=10, rho0::Float64=1.0,
              rhofac::Float64=1.2, rhomax::Float64=10, verbose::Bool=false)

    (; success, total, lalpha, lbeta, fuse_wt) = bb
    rho = rho0

    if !isnothing(start)
        bb.lalpha = start[1]
        bb.lbeta = start[2]
    else
        set_start!(bb)
    end

    for itr in 1:maxiter

        nfail, maxg, avgg = gradstep_fuse(bb; rho=rho)
        if verbose
            println("rho=$(rho)")
            println("$(nfail) failures")
            println("max |G| = $(maxg)")
            println("avg |G| = $(avgg)")
        end

        fuse!(bb)

        rho *= rhofac
        rho = clamp(rho, 0, rhomax)
    end
end

function coef(bb::BetaBinomSeq)
    return bb.lalpha, bb.lbeta
end

function moments(bb::BetaBinomSeq)
    return bb.mean, bb.icc
end

function fit(::Type{BetaBinomSeq}, success::AbstractVector, total::AbstractVector;
             fuse_wt::Float64=1.0, maxiter::Int=100, rho0::Float64=1.0, rhofac::Float64=1.2,
             rhomax::Float64=10.0, verbose::Bool=false, start=nothing)

    if length(success) != length(total)
        error("Length of success $(length(success)) and total $(length(total)) differ.")
    end

    success = Vector{Int32}(success)
    total = Vector{Int32}(total)

    n = length(total)
    bb = BetaBinomSeq(success, total; fuse_wt=fuse_wt)
    fit!(bb; maxiter=maxiter, rho0=rho0, rhofac=rhofac, rhomax=rhomax,
         start=start, verbose=verbose)

    return bb
end

