
function gendat_fuse(rng, s_mean, s_icc; bs1=2, bs2=10, n=1000, tm=100)

    if !all(0 .<= s_mean .<= 1)
        error("invalid s_mean=$(s_mean)")
    end

    if !all(0 .<= s_icc .<= 1)
        error("invalid s_icc=$(s_icc)")
    end

    total = rand(Poisson(tm), n)
    success = zeros(Int, n) # to be filled in below
    bb = BetaBinomSeq(success, total; bs=bs1)

    m = div(n, bs1) # number of mean/icc parameters
    r = div(m, bs2)

    mean = zeros(m)
    icc = zeros(m)

    # i indexes blocks within which the mean and icc are constant
    for i in 1:r
        j = (i-1)*bs2
        mean[j+1:j+bs2] .= s_mean[2 - (i % 2)]
        icc[j+1:j+bs2] .= s_icc[2 - (i % 2)]
    end

    alpha = zeros(m)
    beta = zeros(m)
    for i in 1:m
        alpha[i], beta[i] = BetaBinomialModels.moments_to_shape(mean[i], icc[i])
    end

    for i in 1:n
        j = div(i-1, bs1) + 1
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
    bs = 2

    for lam in [0.0, 1.0, 2.0]
        for bs in [1, 2]
            for k in 1:10
                total = rand(rng, 1:100, bs)
                success = [rand(rng, BetaBinomial(total[i], 1, 1)) for i in 1:bs]
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

@testset "fuse 1 (single block, no fusing needed)" begin

    rng = StableRNG(123)

    n = 1000
    tot = 20
    total = fill(tot, n)


    for (alpha, beta) in [(1, 1), (2, 2), (1, 2)]
        success = [rand(rng, BetaBinomial(tot, alpha, beta)) for _ in 1:n]
        bb = fit(BetaBinomSeq, success, total; maxiter=10, fuse_mean_wt=0.0,
                 fuse_icc_wt=0.0, rho0=0.0, rhofac=1.1, rhomax=20.0, bs=1000,
                 verbose=false)
        logit_mean, logit_icc = coef(bb)
        logit_mean1 = logit(alpha/(alpha+beta))
        logit_icc1 = logit(1/(alpha+beta+1))
        @test isapprox(logit_mean[1], logit_mean1, rtol=0.05, atol=0.05)
        @test isapprox(logit_icc[1], logit_icc1, rtol=0.05, atol=0.05)
    end
end

@testset "fuse 2 (4 iid blocks, no fusing)" begin

    rng = StableRNG(123)

    n = 4000
    tot = 25
    total = fill(tot, n)
    bs = 1000
    m = div(n, bs)

    for (mn,icc) in [(0.5, 0.5), (0.5, 0.1), (0.5, 0.9), (0.1, 0.3)]

        alpha, beta = BetaBinomialModels.moments_to_shape(mn, icc)
        success = [rand(rng, BetaBinomial(tot, alpha, beta)) for _ in 1:n]

        bb = fit(BetaBinomSeq, success, total; maxiter=10, fuse_mean_wt=0.0,
                 fuse_icc_wt=0.0, rho0=0.0, rhofac=1.1, rhomax=20.0, bs=bs, verbose=false)

        logit_mean, logit_icc = coef(bb)
        mean = invlogit.(logit_mean)
        icc = invlogit.(logit_icc)
        mean1 = fill(alpha / (alpha + beta), m)
        icc1 = fill(1 / (1 + alpha + beta), m)
        @test isapprox(mean1, mean, rtol=0.05, atol=0.05)
        @test isapprox(icc1, icc, rtol=0.05, atol=0.05)
    end
end

@testset "fuse 3" begin

    rng = StableRNG(321)

    s_mean = [0.8, 0.2]
    s_icc = [0.2, 0.8]
    bs1 = 1
    bs2 = 20
    n = 4000

    # Get the true moments
    p0, icc0, total, success = gendat_fuse(rng, s_mean, s_icc; n=n, bs1=bs1, bs2=bs2)

    # Estimate the moments
    bb = fit(BetaBinomSeq, success, total; maxiter=20, fuse_mean_wt=1.0,
             fuse_icc_wt=1.0, rho0=0.5, rhofac=1.05, rhomax=2.0, bs=bs1, verbose=false)
    p1, icc1 = coef(bb)

    rr = diff(findall(diff(p1) .!= 0))
    if length(rr) > 0
        println("mean smoothing statistics: ", [mean(rr), median(rr)])
    end

    rr = diff(findall(diff(icc1) .!= 0))
    if length(rr) > 0
        println("icc smoothing statistics: ", [mean(rr), median(rr)])
    end

    # Test mean
    ii1 = isapprox.(p0, s_mean[1])
    p_mean1 = mean(p1[ii1])
    #p_sd1 = std(p[ii1])
    ii2 = isapprox.(p0, s_mean[2])
    p_mean2 = mean(p1[ii2])
    #p_sd2 = std(p[ii2])
    #println("Estimated mean: ", invlogit.([p_mean1, p_mean2]))
    @test isapprox(invlogit(p_mean1), s_mean[1], rtol=0.1, atol=0.1)
    @test isapprox(invlogit(p_mean2), s_mean[2], rtol=0.1, atol=0.1)

    # Test ICC
    ii1 = isapprox.(icc0, s_icc[1])
    icc_mean1 = mean(icc1[ii1])
    ##icc_sd1 = std(icc[lalpha .< 0])
    ii2 = isapprox.(icc0, s_icc[2])
    icc_mean2 = mean(icc1[ii2])
    ##icc_sd2 = std(icc1[lalpha .> 0])
    #println("Estimated ICC: ", invlogit.([icc_mean1, icc_mean2]))
    @test isapprox(invlogit(icc_mean1), s_icc[1], rtol=0.1, atol=0.1)
    @test isapprox(invlogit(icc_mean2), s_icc[2], rtol=0.2, atol=0.2)
end

