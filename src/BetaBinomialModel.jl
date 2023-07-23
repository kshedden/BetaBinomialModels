module BetaBinomialModel

    import StatsAPI: fit, vcov, coef

    export BetaBinomModel, vcov, coef, fit

    export BetaBinomSeq, moments

    include("betabinom.jl")
    include("fuse.jl")
end
