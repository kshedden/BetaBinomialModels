function gendat(rng, alpha, beta)

    bd = Beta(alpha, beta)
    n = 1000
    success, total = zeros(Int, n), zeros(Int, n)

    for i = 1:n
        total[i] = 10 + rand(rng, Poisson(10))
        p = rand(rng, bd)
        success[i] = rand(rng, Binomial(total[i], p))
    end

    return success, total
end

@testset "check_score_hess" begin

    lalpha0, lbeta0 = 0.0, 0.0
    alpha0 = exp(lalpha0)
    beta0 = exp(lbeta0)
    agrad = zeros(2)
    H = zeros(2, 2)
    rng = StableRNG(123)

    for i = 1:10
        for lam in [0.0, 1.0]

            # Check the gradients and Hessians here
            lalpha, lbeta = 2 * rand(rng) - 1, 2 * rand(rng) - 1
            x = [lalpha, lbeta]
            alpha, beta = exp(lalpha), exp(lbeta)

            # Testing data
            success, total = gendat(rng, alpha, beta)
            bb = BetaBinomModel(success, total)

            # Check the gradient
            f = x -> BetaBinomialModel.loglike(bb, x[1], x[2], lalpha0, lbeta0, lam)
            ngrad = grad(central_fdm(5, 1), f, x)[1]
            BetaBinomialModel.score!(bb, agrad, lalpha, lbeta, lalpha0, lbeta0, lam)
            @test isapprox(ngrad, agrad, rtol = 1e-7, atol = 1e-7)

            # Check the diagonal of the Hessian
            nhess = zeros(2, 2)
            f1 = y -> grad(central_fdm(5, 2), z -> f([z, x[2]]), y)[1]
            nhess[1, 1] = f1(x[1])
            f2 = y -> grad(central_fdm(5, 2), z -> f([x[1], z]), y)[1]
            nhess[2, 2] = f2(x[2])

            # Check the mixed partial
            f1 = y -> grad(central_fdm(5, 1), z -> f([z, y]), x[1])[1]
            nhess[1, 2] = grad(central_fdm(5, 1), f1, x[2])[1]
            nhess[2, 1] = nhess[1, 2]
            BetaBinomialModel.hess!(bb, agrad, H, lalpha, lbeta, lalpha0, lbeta0, lam)
            @test isapprox(H, nhess, rtol = 1e-5, atol = 1e-5)
        end
    end
end

@testset "check_fit" begin

    lalpha0, lbeta0 = 0.0, 0.0
    lam = 0.0
    nrep = 100
    za, zb = Float64[], Float64[]
    rng = StableRNG(123)

    for i = 1:nrep
        lalpha, lbeta = 2 * rand(rng) - 1, 2 * rand(rng) - 1
        alpha, beta = exp(lalpha), exp(lbeta)
        success, total = gendat(rng, alpha, beta)
        bb = fit(BetaBinomModel, success, total; lalpha0=lalpha0, lbeta0=lbeta0, lam=lam)
        cc = coef(bb)
        vc = vcov(bb)

        push!(za, (cc[1] - lalpha) / sqrt(vc[1, 1]))
        push!(zb, (cc[2] - lbeta) / sqrt(vc[2, 2]))
    end

    @test isapprox(mean(za), 0.0, atol=0.2, rtol=0.15)
    @test isapprox(mean(zb), 0.0, atol=0.2, rtol=0.15)
    @test isapprox(std(za), 1.0, atol=0.2, rtol=0.1)
    @test isapprox(std(zb), 1.0, atol=0.2, rtol=0.1)
end
