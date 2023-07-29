module BetaBinomialModels

    import StatsAPI: fit, vcov, coef

    export BetaBinomModel, vcov, coef, fit

    export BetaBinomSeq

    include("betabinom.jl")
    include("fuse.jl")
end
