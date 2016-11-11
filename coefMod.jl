module coefMod

export CEOpt_


include("runexp.jl")
include("ceopt.jl")


function drawSample(p)

    return rand(Truncated(Normal(p[1], p[2]), 0, Inf))
end

function computePerf_(rewards, n, N, x)

    regret_acc, played_acc, nBestArm = runExpN(rewards, Any[:UCB1, x], n = n, N = N)

    return -regret_acc[end]
end

computePerf(rewards, n, N) = (x) -> computePerf_(rewards, n, N, x)

function updateParam(X, S, gamma_)

    I = map((x) -> x >= gamma_ ? 1 : 0, S)

    p = Array(Float64, 2)
    p[1] = sum(I .* X) / sum(I)
    p[2]= sqrt(sum(I .* (X - p[1]).^2) / sum(I))

    return p
end


function CEOpt_(param)

    rewards = [RareDist(param[1], param[2], Truncated(Normal(param[3], param[4]), param[3] - 5 * param[4], min(param[3] + 5 * param[4], 0))), RareDist(param[5], param[6], Truncated(Normal(param[7], param[8]), param[7] - 5 * param[8], min(param[7] + 5 * param[8], 0)))]

    p_init_mean = max(abs(mean(rewards[1])), abs(mean(rewards[2])))
    p_init_sigma = max(abs(param[2] / 10), abs(param[6] / 10))
    p_init = [p_init_mean, p_init_sigma]

    exp_n = 10000
    exp_N = 1

    N = 100
    rho = log(N) / N

    p = CEOpt(drawSample, p_init, computePerf(rewards, exp_n, exp_N), updateParam, N, rho)

    return p[1]
end

end


