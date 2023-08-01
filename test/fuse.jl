
function gendat_fuse(rng, s_mean, s_icc; bs=10, n=1000, tm=100)

    if !all(0 .<= s_mean .<= 1)
        error("invalid s_mean=$(s_mean)")
    end

    if !all(0 .<= s_icc .<= 1)
        error("invalid s_icc=$(s_icc)")
    end

    total = rand(Poisson(tm), n)
    success = zeros(Int, n) # to be filled in below
    bb = BetaBinomSeq(success, total)

    m = div(n, bs) # number of mean/icc parameters

    mean = zeros(m)
    icc = zeros(m)

    # i indexes blocks within which the mean and icc are constant
    for i in 1:m
        mean[i] = s_mean[2 - (i % 2)]
        icc[i] = s_icc[2 - (i % 2)]
    end

    alpha = zeros(m)
    beta = zeros(m)
    for i in 1:m
        alpha[i], beta[i] = BetaBinomialModels.moments_to_shape(mean[i], icc[i])
    end

    for i in 1:n
        j = div(i-1, bs) + 1
        success[i] = rand(rng, BetaBinomial(total[i], alpha[j], beta[j]))
    end

    return mean, icc, total, success
end

logit(x) = log(x / (1 - x))
invlogit(x) = 1 / (1 + exp(-x))

@testset "fuse likelihood and gradient" begin

    rng = StableRNG(123)
    agrad = zeros(2)
    lam = 1.0

    for lam in [0.0, 1.0, 2.0]
        for k in 1:10
            total = rand(rng, 1:100)
            success = rand(rng, BetaBinomial(total, 1, 1))
            x = randn(rng, 2)
            logit_mean0 = randn(rng)
            logit_icc0 = randn(rng)
            f = par -> BetaBinomialModels.loglike_single(success, total, logit_mean0, logit_icc0, par, lam)
            ngrad = grad(central_fdm(5, 1), f, x)[1]
            BetaBinomialModels.score_single!(success, total, logit_mean0, logit_icc0, x, agrad, lam)
            @test isapprox(ngrad, agrad)
        end
    end
end

@testset "fuse moments roundtrip" begin

    rng = StableRNG(321)

    for j in 1:10
        alpha = rand(rng, Gamma(1, 1))
        beta = rand(rng, Gamma(1, 1))
        alpha0, beta0 = alpha, beta
        mean, icc = 0.0, 0.0
        for k in 1:2
            mean, icc = BetaBinomialModels.shape_to_moments(alpha, beta)
            alpha, beta = BetaBinomialModels.moments_to_shape(mean, icc)
        end
        @test isapprox(alpha, alpha0)
        @test isapprox(beta, beta0)
    end
end

@testset "fuse 1 (single block, fully fused)" begin

    rng = StableRNG(123)

    n = 2000
    tot = 100
    total = fill(tot, n)

    for (alpha, beta) in [(1, 1), (2, 2), (1, 2)]
        success = [rand(rng, BetaBinomial(tot, alpha, beta)) for _ in 1:n]
        bb = fit(BetaBinomSeq, success, total; maxiter=10, fuse_mean_wt=10.0,
                 fuse_icc_wt=10.0, rho0=0.5, rhofac=1.1, rhomax=10.0,
                 verbose=false)
        logit_mean, logit_icc = coef(bb)
        mn = invlogit.(logit_mean)
        icc = invlogit.(logit_icc)
        mn1 = alpha / (alpha + beta)
        icc1 = 1 / (alpha + beta + 1)
        @test isapprox(mean(mn), mn1, rtol=0.05, atol=0.05)
        @test isapprox(mean(icc), icc1, rtol=0.08, atol=0.08)
    end
end

@testset "fuse blocks of size 20" begin

    rng = StableRNG(321)

    s_mean = [0.8, 0.2]
    s_icc = [0.2, 0.8]
    bs = 20
    n = 4000

    # Get the true moments
    p0, icc0, total, success = gendat_fuse(rng, s_mean, s_icc; n=n, bs=bs)

    # Estimate the moments
    bb = fit(BetaBinomSeq, success, total; maxiter=20, fuse_mean_wt=1.0,
             fuse_icc_wt=1.0, rho0=0.5, rhofac=1.05, rhomax=2.0, verbose=false)
    p1, icc1 = coef(bb)

    # Test mean
    ii1 = isapprox.(kron(p0, ones(bs)), s_mean[1])
    p_mean1 = mean(p1[ii1])
    ii2 = isapprox.(kron(p0, ones(bs)), s_mean[2])
    p_mean2 = mean(p1[ii2])
    @test isapprox(invlogit(p_mean1), s_mean[1], rtol=0.1, atol=0.1)
    @test isapprox(invlogit(p_mean2), s_mean[2], rtol=0.1, atol=0.1)

    # Test ICC
    ii1 = isapprox.(kron(icc0, ones(bs)), s_icc[1])
    icc_mean1 = mean(icc1[ii1])
    ii2 = isapprox.(kron(icc0, ones(bs)), s_icc[2])
    icc_mean2 = mean(icc1[ii2])
    @test isapprox(invlogit(icc_mean1), s_icc[1], rtol=0.1, atol=0.1)
    @test isapprox(invlogit(icc_mean2), s_icc[2], rtol=0.2, atol=0.2)
end
