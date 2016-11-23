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


rewards = [RareDist(0.01, -1000., Truncated(Normal(-30, 4), -50, -10)), RareDist(0., -1000., Truncated(Normal(-70, 10), -120, -20))]

N = 100
rho = log(N) / N

p = CEOpt(drawSample, [150, 150], computePerf(rewards, 10000, 100), updateParam, N, rho, debug = 1)

println(p)


