using Lasso

mutable struct BetaBinomSeq{T<:Integer}

    # Number of successes per observation
    success::Vector{T}

    # Total number of trials per observation
    total::Vector{T}

    # An indicator that the MLE optimization converged
    converged::Bool

    bs::Int64

    # The penalty parameter for the fused lasso
    fuse_mean_wt
    fuse_icc_wt

    # The mean and ICC corresponding to the shape parameters, on the logit scale
    logit_mean::Vector{Float64}
    logit_icc::Vector{Float64}

    # The ADMM u vectors
    logit_mean_u::Vector{Float64}
    logit_icc_u::Vector{Float64}
end

function BetaBinomSeq(success::AbstractVector, total::AbstractVector;
                      fuse_mean_wt=0.0, fuse_icc_wt=0.0, bs=2)
    @assert length(success) == length(total)
    n = length(total)
    m = div(n, bs)
    logit_mean = zeros(m)
    logit_icc = zeros(m)
    logit_mean_u = zeros(m)
    logit_icc_u = zeros(m)

    return BetaBinomSeq(success, total, false, bs, fuse_mean_wt, fuse_icc_wt,
                        logit_mean, logit_icc, logit_mean_u, logit_icc_u)
end

logit(x) = log(x / (1 - x))

invlogit(x) = 1 / (1 + exp(-x))

function loglike_single(
    success::Vector{T},
    total::Vector{T},
    logit_mean0::Float64,
    logit_icc0::Float64,
    par::Vector{Float64},
    rho::Float64,
) where {T<:Integer}
    logit_mean, logit_icc = par[1], par[2]
    mean, icc = invlogit(logit_mean), invlogit(logit_icc)
    alpha, beta = moments_to_shape(mean, icc)

    e = 0.01
    if mean < e || mean > 1-e || icc < e || icc > 1-e
        return -Inf
    end

    ll = 0.0
    for i in eachindex(success)
        ll += logabsgamma(success[i] + alpha)[1] + logabsgamma(total[i] - success[i] + beta)[1] +
                logabsgamma(alpha + beta)[1]
        ll -= logabsgamma(total[i] + alpha + beta)[1] + logabsgamma(alpha)[1] + logabsgamma(beta)[1]
    end

    ll -= rho * (par[1] - logit_mean0)^2 / 2
    ll -= rho * (par[2] - logit_icc0)^2 / 2

    ee = 0.1 # TODO how to set this?
    ll -= ee * par[1]^2 / 2
    ll -= ee * par[2]^2 / 2

    return ll
end

function score_single!(
    success::Vector{T},
    total::Vector{T},
    logit_mean0::Float64,
    logit_icc0::Float64,
    par::Vector{Float64},
    G::Vector{Float64},
    rho::Float64,
) where {T<:Integer}

    logit_mean, logit_icc = par[1], par[2]
    mean, icc = invlogit(logit_mean), invlogit(logit_icc)
    alpha, beta = moments_to_shape(mean, icc)

    e = 0.01
    if mean < e || mean > 1-e || icc < e || icc > 1-e
        G .= NaN
        return
    end

    # partial L / partial (alpha, beta)
    G .= 0
    for i in eachindex(total)
        dab = digamma(alpha + beta)
        dnab = digamma(total[i] + alpha + beta)
        G[1] += digamma(success[i] + alpha) + dab - dnab - digamma(alpha)
        G[2] += digamma(total[i] - success[i] + beta) + dab - dnab - digamma(beta)
    end

    # partial (m, i) / partial (alpha, beta)
    H = zeros(2, 2)
    u = (alpha + beta)^2
    H[1, 1] = beta / u
    H[1, 2] = -alpha / u
    H[2, 1] = -1 / (alpha + beta + 1)^2
    H[2, 2] = H[2, 1]

    e1 = exp(par[1])
    e2 = exp(par[2])
    Q = Diagonal([e1 / (1 + e1)^2, e2 / (1 + e2)^2])

    # Adjust for the change of variables
    G .= Q * (H' \ G)

    G[1] -= rho * (par[1] - logit_mean0)
    G[2] -= rho * (par[2] - logit_icc0)

    ee = 0.1
    G[1] -= ee * par[1]
    G[2] -= ee * par[2]
end

function admm!(success::Vector{T}, total::Vector{T}, logit_mean::Float64, logit_icc::Float64,
               par::Vector{Float64}, par1::Vector{Float64}, G::Vector{Float64},
               rho::Float64; step0::Float64=1.0, gtol::Float64=1e-5) where {T<:Integer}

    cnvrg = false
    gnrm = 0.0
    for k in 1:20 # TODO make this configurable
        cnvrg, gnrm = admm_step!(success, total, logit_mean, logit_icc, par, par1, G, rho;
                                 step0=step0, gtol=gtol)
    end

    return cnvrg, gnrm
end

function admm_step!(success::Vector{T}, total::Vector{T}, logit_mean0::Float64, logit_icc0::Float64,
                    par::Vector{Float64}, par1::Vector{Float64}, G::Vector{Float64},
                    rho::Float64; step0::Float64=1.0, gtol::Float64=1e-5) where {T<:Integer}

    f0 = loglike_single(success, total, logit_mean0, logit_icc0, par, rho)
    score_single!(success, total, logit_mean0, logit_icc0, par, G, rho)

    if norm(G) < gtol
        return true, norm(G)
    end

    while step0 >= 1e-20
        par1 .= par + step0*G
        f1 = loglike_single(success, total, logit_mean0, logit_icc0, par1, rho)
        if f1 > f0
            par .= par1
            return true, norm(G)
        end
        step0 /= 2
    end
    return false, norm(G)
end

function gradstep_fuse(bb::BetaBinomSeq; step0::Float64=1.0, rho::Float64=1.0,
                       gtol::Float64=1e-5)

    (; success, total, logit_mean, logit_icc, logit_mean_u, logit_icc_u, bs) = bb

    n = length(total)
    G = zeros(2)
    lam = 1.0
    par = zeros(2)
    par1 = zeros(2)
    nfail = 0
    maxg, avgg = 0.0, 0.0
    m = div(n, bs)

    logit_mean0 = logit_mean - logit_mean_u
    logit_icc0 = logit_icc - logit_icc_u

    for i in 1:div(n, bs)

        par[1] = logit_mean[i]
        par[2] = logit_icc[i]
        i1 = (i-1)*bs + 1
        i2 = i*bs
        cnvrg, gnrm = admm!(success[i1:i2], total[i1:i2], logit_mean0[i], logit_icc0[i],
                            par, par1, G, rho; step0=step0, gtol=gtol)
        if cnvrg
            logit_mean[i], logit_icc[i] = par[1], par[2]
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

    (; success, total, logit_mean, logit_icc, bs) = bb

    logit_icc .= logit(0.5)

    for j in eachindex(logit_mean)
        i1 = (j-1)*bs + 1
        i2 = j*bs
        mm = (sum(success[i1:i2]) + 1) / (sum(total[i1:i2]) + 2)
        mm = clamp.(mm, 0.1, 0.9)
        logit_mean[j] = logit(mm)
    end
end

function shape_to_moments(alpha::Float64, beta::Float64)
    mean = alpha / (alpha + beta)
    icc = 1 / (alpha + beta + 1)
    return mean, icc
end

function moments_to_shape(mean::Float64, icc::Float64)
    r = mean / (1 - mean)
    alpha = ((1 - icc) / icc) * (r / (1 + r))
    beta = (1 - (icc + r*(1 - icc) / (1 + r))) / icc
    return alpha, beta
end

function fuse!(bb::BetaBinomSeq, rho::Float64)

    (; logit_mean, logit_icc, logit_mean_u, logit_icc_u, fuse_mean_wt, fuse_icc_wt) = bb

    d1 = 0.0
    if fuse_mean_wt > 0
        r = fit(FusedLasso, logit_mean + logit_mean_u, fuse_mean_wt / rho)
        logit_mean1 = coef(r)
        logit_mean_u .+= logit_mean - logit_mean1
        d1 = sqrt(sum(abs2, logit_mean1 - logit_mean))
        logit_mean .= logit_mean1
    end

    d2 = 0.0
    if fuse_icc_wt > 0
        r = fit(FusedLasso, logit_icc + logit_icc_u, fuse_icc_wt / rho)
        logit_icc1 = coef(r)
        logit_icc_u .+= logit_icc - logit_icc1
        d2 = sqrt(sum(abs2, logit_icc1 - logit_icc))
        logit_icc .= logit_icc1
    end
end

function fit!(bb::BetaBinomSeq; start=nothing, maxiter::Int=10, rho0::Float64=1.0,
              rhofac::Float64=1.2, rhomax::Float64=10, gtol::Float64=1e-5,
              step0::Float64=0.1, verbose::Bool=false)

    (; success, total, logit_mean, logit_icc, fuse_mean_wt, fuse_icc_wt) = bb
    rho = rho0

    for itr in 1:maxiter

        nfail, maxg, avgg = gradstep_fuse(bb; rho=rho, step0=step0, gtol=gtol)
        if verbose
            println("rho=$(rho)")
            println("$(nfail) failures")
            println("max |G| = $(maxg)")
            println("avg |G| = $(avgg)")
        end

        fuse!(bb, rho)

        rho *= rhofac
        rho = clamp(rho, 0, rhomax)
    end
end

function coef(bb::BetaBinomSeq)
    return bb.logit_mean, bb.logit_icc
end

function fit(::Type{BetaBinomSeq}, success::AbstractVector, total::AbstractVector;
             fuse_mean_wt::Real=1.0, fuse_icc_wt::Real=1.0, maxiter::Int=100, bs::Int=2,
             rho0::Float64=1.0, rhofac::Float64=1.2, rhomax::Float64=10.0, gtol::Float64=1e-5,
             step0::Float64=1.0, verbose::Bool=false, start=nothing, dofit::Bool=true)

    if length(success) != length(total)
        error("Length of success $(length(success)) and total $(length(total)) differ.")
    end

    if length(success) % bs != 0
        error("Block size must evenly divide the sequence length")
    end

    success = Vector{Int32}(success)
    total = Vector{Int32}(total)

    n = length(total)
    bb = BetaBinomSeq(success, total; fuse_mean_wt=fuse_mean_wt, fuse_icc_wt=fuse_icc_wt, bs=bs)

    if !isnothing(start)
        m = div(n, bs)
        if (length(start[1]) != m) || (length(start[2]) != m)
            error("Starting vectors have invalid lengths ($(length(start[1])), $(length(start[2])) != $m)")
        end
        bb.logit_mean = copy(start[1])
        bb.logit_icc = copy(start[2])
    else
        set_start!(bb)
    end

    if dofit
        fit!(bb; maxiter=maxiter, rho0=rho0, rhofac=rhofac, rhomax=rhomax,
             start=start, verbose=verbose)
    end

    return bb
end

