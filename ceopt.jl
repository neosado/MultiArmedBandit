function CEOpt(sample::Function, p0::Any, perf::Function, update::Function, N::Int64, rho::Float64; debug::Int64 = 0, bSmooth::Bool = false, alpha::Float64 = 0.7)

    p = p0

    X = Array(Any, N)
    S = Array(Float64, N)

    gamma_prev = 0
    d = 0
    p_prev = p0

    t = 1

    while d < 3
        for i = 1:N
            X[i] = sample(p)
            S[i] = perf(X[i])
        end

        Ssorted = sort(S)
        gamma_ = Ssorted[round(Int64, (1 - rho) * N)]

        if bSmooth
            w = update(X, S, gamma_)
            p = alpha * w + (1 - alpha) * p_prev
        else
            p = update(X, S, gamma_)
        end

        if debug > 0
            if debug > 1
                println("X: ", X)
                println("S: ", S)
            end
            println(t, ": ", gamma_, ", ", p)
        end

        #if abs((gamma_ - gamma_prev) / gamma_prev) < 0.01
        # XXX for MAB
        if abs((p[1] - p_prev[1]) / p_prev[1]) < 0.1
            d += 1
        else
            d = 0
        end

        gamma_prev = gamma_
        p_prev = p

        t += 1
    end

    return round(Int64, p)
end


if false
    using Distributions


    function drawSample01(p)

        r = zeros(Int64, length(p))

        for i = 1:length(p)
            r[i] = rand(Bernoulli(p[i]))
        end

        return r
    end

    function computePerf01(x)

        y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        S = length(x) - sum(abs(x - y))

        return S
    end

    function updateParam01(X, S, gamma_)

        I = map((x) -> x >= gamma_ ? 1 : 0, S)

        p = sum(I .* X) / sum(I)

        return p
    end


    p = CEOpt(drawSample01, 1/2 * ones(10), computePerf01, updateParam01, 50, 0.1)

    println(p)
end


