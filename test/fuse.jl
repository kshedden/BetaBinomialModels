
function gendat_fuse(rng, s)

    n = 1000
    total = zeros(Int, n)
    success = zeros(Int, n)
    lalpha = zeros(n)
    lbeta = zeros(n)

    m = div(n, 10)
    for i in 1:m
        j = (i-1)*10
        lalpha[j+1:j+10] .= (i % 2 == 1) ? s : -s
        lbeta[j+1:j+10] .= (i % 2 == 1) ? -s : s
    end

    for i in 1:n
        alpha = exp(lalpha[i])
        beta = exp(lbeta[i])
        total[i] = rand(rng, Poisson(20))
        success[i] = rand(rng, BetaBinomial(total[i], alpha, beta))
    end

    return lalpha, lbeta, total, success
end

@testset "fuse likelihood and gradient" begin

    rng = StableRNG(123)
    agrad = zeros(2)
    lalpha0 = 0.0
    lbeta0 = 0.0
    lam = 1.0

    for k in 1:10
        total = rand(rng, 1:100)
        success = rand(rng, 0:total)
        x = randn(rng, 2)
        f = par -> BetaBinomialModel.loglike_single(success, total, lalpha0, lbeta0, par, lam)
        ngrad = grad(central_fdm(5, 1), f, x)[1]
        BetaBinomialModel.score_single!(success, total, lalpha0, lbeta0, x, agrad, lam)
        @test isapprox(ngrad, agrad)
    end
end

@testset "fuse moments roundtrip" begin

    rng = StableRNG(321)

    lalpha, lbeta, total, success = gendat_fuse(rng, 0.5)

    bb = fit(BetaBinomSeq, success, total; maxiter=0, verbose=false)

    n = length(bb.total)
    lalpha = randn(rng, n)
    lbeta = randn(rng, n)
    bb.lalpha = copy(lalpha)
    bb.lbeta = copy(lbeta)
    for k in 1:2
        BetaBinomialModel.get_mean_icc!(bb)
        BetaBinomialModel.reset_shape!(bb)
    end

    @test isapprox(bb.lalpha, lalpha)
    @test isapprox(bb.lbeta, lbeta)
end

@testset "fuse 1" begin

    rng = StableRNG(321)

    alpha = 1.0
    beta = 1.0
    n = 1000
    tot = 20
    total = fill(tot, n)
    success = [rand(rng, BetaBinomial(tot, alpha, beta)) for _ in 1:n]

    bb = fit(BetaBinomSeq, success, total; maxiter=50, fuse_wt=10.0, rho0=1.0,
               rhofac=1.1, rhomax=20.0, verbose=false)
    lalpha1, lbeta1 = coef(bb)
    alpha1 = exp.(lalpha1)
    beta1 = exp.(lbeta1)
    p, icc = moments(bb)

    @test isapprox(mean(p), alpha/(alpha+beta), rtol=0.1, atol=0.03)
    @test isapprox(mean(icc), 1/(alpha+beta+1), rtol=0.1, atol=0.03)
    @test isapprox(std(icc), 0, rtol=0.1, atol=0.01)
end

@testset "fuse 2" begin

    rng = StableRNG(321)

    for s in [0.2, 0.5 ,0.7]
        lalpha, lbeta, total, success = gendat_fuse(rng, s)

        bb = fit(BetaBinomSeq, success, total; maxiter=50, fuse_wt=5.0, rho0=1.0,
                 rhofac=1.1, rhomax=50.0, verbose=false)
        lalpha1, lbeta1 = coef(bb)
        p, icc = moments(bb)

        # True moments
        a = exp.(lalpha)
        b = exp.(lbeta)
        p0 = a ./ (a + b)
        icc0 = 1 ./ (a + b .+ 1)

        # Test mean
        m1 = mean(p[lalpha .< 0])
        sd1 = std(p[lalpha .< 0])
        m2 = mean(p[lalpha .> 0])
        sd2 = std(p[lalpha .> 0])
        m1x = mean(p0[lalpha .< 0])
        m2x = mean(p0[lalpha .> 0])
        @test isapprox(m1, m1x, rtol=0.1, atol=0.1)
        @test isapprox(m2, m2x, rtol=0.1, atol=0.1)

        # Test ICC
        m1 = mean(icc[lalpha .< 0])
        sd1 = std(icc[lalpha .< 0])
        m2 = mean(icc[lalpha .> 0])
        sd2 = std(icc[lalpha .> 0])
        m1x = mean(icc0[lalpha .< 0])
        m2x = mean(icc0[lalpha .> 0])
        @test isapprox(m1, m1x, rtol=0.1, atol=0.1)
        @test isapprox(m2, m2x, rtol=0.1, atol=0.1)
    end
end


