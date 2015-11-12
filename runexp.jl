using Distributions
using ConjugatePriors
using PyPlot
using Base.Test

import Base: rand, mean, maximum, minimum, string
import ConjugatePriors: NormalGamma

push!(LOAD_PATH, ".")
using Util


type RareDist
    
    p::Float64
    r_type::Symbol
    r::Float64
    Dr::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution}
    D::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution}

    function RareDist(p::Float64, r::Float64, D::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution})
        
        self = new()
        
        self.p = p
        self.r_type = :real
        self.r = r
        self.D = D
        
        return self
    end

    function RareDist(p::Float64, Dr::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution}, D::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution})
        
        self = new()
        
        self.p = p
        self.r_type = :dist
        self.Dr = Dr
        self.D = D
        
        return self
    end
end

function rand(D::RareDist)
    
    u = rand()
    
    if u < D.p
        if D.r_type == :real
            return D.r
        elseif D.r_type == :dist
            return rand(D.Dr)
        end
    else
        return rand(D.D)
    end
end

function mean(D::RareDist)

    if D.r_type == :real
        return D.p * D.r + (1 - D.p) * mean(D.D)
    elseif D.r_type == :dist
        return D.p * mean(D.Dr) + (1 - D.p) * mean(D.D)
    end
end

function maximum(D::RareDist)

    if D.p == 0.
        return maximum(D.D)
    else
        if D.r_type == :real
            return max(D.r, maximum(D.D))
        elseif D.r_type == :dist
            return max(maximum(D.Dr), maximum(D.D))
        end
    end
end

function minimum(D::RareDist)

    if D.p == 0.
        return minimum(D.D)
    else
        if D.r_type == :real
            return min(D.r, minimum(D.D))
        elseif D.r_type == :dist
            return min(minimum(D.Dr), minimum(D.D))
        end
    end
end

function string(D::RareDist)

    if D.p == 0.
        return string(D.D)
    else
        if D.r_type == :real
            return string(D.p) * ", " * string(D.r) * "; " * string(1 - D.p) * ", " * string(D.D)
        elseif D.r_type == :dist
            return string(D.p) * ", " * string(D.Dr) * "; " * string(1 - D.p) * ", " * string(D.D)
        end
    end
end


type UCBInt
    
    rewards
    
    nArms::Int64
    
    N_total::Int64
    N::Vector{Int64}
    Q::Vector{Float64}
    
    function UCBInt(rewards)
        
        self = new()
        
        self.rewards = rewards
        
        self.nArms = length(rewards)
        
        self.N_total = 0
        self.N = zeros(Int64, self.nArms)
        self.Q = zeros(self.nArms)
        
        return self
    end
end


function UCB(ucb::UCBInt, c::Float64)

    Qv = zeros(ucb.nArms)
    
    for a_ = 1:ucb.nArms
        if ucb.N[a_] == 0
            Qv[a_] = Inf
        else
            Qv[a_] = ucb.Q[a_] + c * sqrt(log(ucb.N_total) / ucb.N[a_])
        end
    end
    
    a = argmax(Qv)
    
    q = rand(ucb.rewards[a])
    
    ucb.N_total += 1
    ucb.N[a] += 1
    ucb.Q[a] += (q - ucb.Q[a]) / ucb.N[a]

    return a, q
end


function A_UCB(rewards, params::Vector{Float64}; n::Int64 = 10000, policy = nothing, bPlot::Bool = false, plotfunc::Symbol = :semilogx, verbose::Int64 = 1)

    nArms = length(rewards)

    N_total = 0
    N = zeros(Int64, nArms)
    Q = zeros(nArms)

    N_hist = zeros(Int64, nArms, n)
    Q_hist = zeros(nArms, n)
    Qv_hist = zeros(nArms, n)

    nParams = length(params)

    ucb = Array(UCBInt, nParams)

    for i = 1:nParams
        ucb[i] = UCBInt(rewards)
    end

    Nc_total = 0
    Nc = zeros(Int64, nParams)
    Qc = zeros(nParams)

    Nc_hist = zeros(Int64, nParams, n)
    Qc_hist = zeros(nParams, n)

    if policy == nothing
        policy = [:UCB1, sqrt(2)]
    end

    if policy[1] == :Exp3
        K = nParams

        g = n
        gam = min(1, sqrt(K * log(K) / ((e - 1) * g)))

        w = ones(K)
        p = zeros(K)
    end

    for i = 1:n
        if policy[1] == :UCB1
            Qv = zeros(nParams)

            for j = 1:nParams
                if Nc[j] == 0
                    Qv[j] = Inf
                else
                    c = policy[2]
                    Qv[j] = Qc[j] + c * sqrt(log(Nc_total) / Nc[j])
                end
            end

            j_ = argmax(Qv)

        elseif policy[1] == :Exp3
            sum_w = sum(w)

            for j = 1:K
                p[j] = (1 - gam) * w[j] / sum_w + gam / K
            end

            j_ = sampleFromProb(p)

        end

        a, q = UCB(ucb[j_], params[j_])

        if policy[1] == :Exp3
            w[j_] *= exp(gam * q / p[j_] / K)
        end

        for a_ = 1:nArms
            N[a_] = sum([ucb[j].N[a_] for j = 1:nParams])
            if N[a_] > 0
                Q[a_] = sum([ucb[j].Q[a_] * ucb[j].N[a_] for j = 1:nParams]) / N[a_]
            end
            N_hist[a_, i] = N[a_]
            Q_hist[a_, i] = Q[a_]
            Qv_hist[a_, i] = Qv[a_]
        end

        Nc_total += 1
        Nc[j_] += 1
        Qc[j_] += (q - Qc[j_]) / Nc[j_]

        for j = 1:nParams
            Nc_hist[j, i] = Nc[j]
            Qc_hist[j, i] = Qc[j]
        end
    end

    U = map(mean, rewards)
    bestArm = argmax(U)
    
    Regret = zeros(n)
    Played = zeros(n)
    c_hist = zeros(nParams, n)

    for i = 1:n
        Regret[i] = dot((U[bestArm] - U), N_hist[:, i])
        Played[i] = N_hist[bestArm, i] / i
        c_hist[:, i] = Nc_hist[:, i] / i
    end
    
    if bPlot
        figure()
        for j = 1:nParams
            eval(plotfunc)(1:n, vec(c_hist[j, :]) * 100)
        end
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of c played")
        legend([string(round(Int64, params[i])) for i = 1:nParams], loc = "best")

        figure()
        for j = 1:nParams
            eval(plotfunc)(1:n, vec(Qc_hist[j, :]))
        end
        xlabel("Number of plays")
        ylabel("Qc")
        legend([string(round(Int64, params[i])) for i = 1:nParams], loc = "best")

        if verbose > 0
            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(N_hist[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("N")
            legend([string(i) for i = 1:nArms], loc = "best")
        end

        figure()
        for a_ = 1:nArms
            eval(plotfunc)(1:n, vec(Q_hist[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("Q")
        legend([string(i) for i = 1:nArms], loc = "best")

        if verbose > 0
            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(Qv_hist[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("Qv")
            legend([string(i) for i = 1:nArms], loc = "best")
        end

        figure()
        eval(plotfunc)(1:n, Regret)
        xlabel("Number of plays")
        ylabel("Regret")

        figure()
        eval(plotfunc)(1:n, Played * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best arm played")
    end

    return N_total, N, N_hist, Q, Q_hist, Qv_hist, Regret, Played, c_hist, Qc_hist
end


function ThompsonSampling(rewards; n::Int64 = 10000, bPlot::Bool = false, plotfunc::Symbol = :semilogx, verbose::Int64 = 1)

    nArms = length(rewards)

    N_total = 0
    N = zeros(Int64, nArms)
    Q = zeros(nArms)

    N_hist = zeros(Int64, nArms, n)
    Q_hist = zeros(nArms, n)
    Qv_hist = zeros(nArms, n)

    S = zeros(Int64, nArms)
    F = zeros(Int64, nArms)
    S_hist = zeros(nArms, n)

    theta = zeros(nArms)

    min_bound = minimum([minimum(rewards[i]) for i = 1:nArms])
    max_bound = maximum([maximum(rewards[i]) for i = 1:nArms])

    for i = 1:n
        for j = 1:nArms
            theta[j] = rand(Beta(S[j] + 1, F[j] + 1))
        end

        k = argmax(theta)

        q = rand(rewards[k])
        q_ = (q - min_bound) / (max_bound - min_bound)

        if rand(Bernoulli(q_)) == 1
            S[k] += 1
        else
            F[k] += 1
        end

        N_total += 1
        N[k] += 1
        Q[k] += (q - Q[k]) / N[k]

        for j = 1:nArms
            N_hist[j, i] = N[j]
            Q_hist[j, i] = Q[j]
            Qv_hist[j, i] = theta[j]
            if S[j] + F[j] > 0
                S_hist[j, i] = S[j] / (S[j] + F[j])
            end
        end
    end

    U = map(mean, rewards)
    bestArm = argmax(U)
    
    Regret = zeros(n)
    Played = zeros(n)

    for i = 1:n
        Regret[i] = dot((U[bestArm] - U), N_hist[:, i])
        Played[i] = N_hist[bestArm, i] / i
    end
    
    if bPlot
        figure()
        for a_ = 1:nArms
            eval(plotfunc)(1:n, vec(S_hist[a_, :]))
        end
        ylim([0, 1])
        xlabel("Number of plays")
        ylabel("Theta")
        legend([string(i) for i = 1:nArms], loc = "best")

        if verbose > 0
            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(N_hist[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("N")
            legend([string(i) for i = 1:nArms], loc = "best")
        end

        figure()
        for a_ = 1:nArms
            eval(plotfunc)(1:n, vec(Q_hist[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("Q")
        legend([string(i) for i = 1:nArms], loc = "best")

        if verbose > 0
            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(Qv_hist[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("Qv")
            legend([string(i) for i = 1:nArms], loc = "best")
        end

        figure()
        eval(plotfunc)(1:n, Regret)
        xlabel("Number of plays")
        ylabel("Regret")

        figure()
        eval(plotfunc)(1:n, Played * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best arm played")
    end
    
    return N_total, N, N_hist, Q, Q_hist, Qv_hist, Regret, Played, S_hist
end


type ArmModel

    N::Int64

    V_alpha0::Int64
    V_beta0::Int64
    V::Int64

    mu0::Float64        # mu_0
    lambda0::Float64    # n_mu
    alpha0::Float64     # n_tau / 2
    beta0::Float64      # n_tau / (2 * tau0)

    mu::Float64
    lambda::Float64
    alpha::Float64
    beta_::Float64

    n::Int64
    q::Float64
    x2::Float64

    v_bound::Float64

    mu_v0::Float64      # mu_0
    lambda_v0::Float64  # n_mu
    alpha_v0::Float64   # n_tau / 2
    beta_v0::Float64    # n_tau / (2 * tau0)

    mu_v::Float64
    lambda_v::Float64
    alpha_v::Float64
    beta_v::Float64

    n_v::Int64
    q_v::Float64
    x2_v::Float64


    function ArmModel(V_alpha0::Int64, V_beta0::Int64, mu0::Float64, lambda0::Float64, alpha0::Float64, beta0::Float64, v_bound::Float64, mu_v0::Float64, lambda_v0::Float64, alpha_v0::Float64, beta_v0::Float64)

        self = new()

        self.N = 0

        self.V_alpha0 = V_alpha0
        self.V_beta0 = V_beta0
        self.V = 0

        self.mu0 = mu0
        self.lambda0 = lambda0
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.mu = mu0
        self.lambda = lambda0
        self.alpha = alpha0
        self.beta_ = beta0

        self.n = 0
        self.q = 0
        self.x2 = 0

        self.v_bound = v_bound

        self.mu_v0 = mu_v0
        self.lambda_v0 = lambda_v0
        self.alpha_v0 = alpha_v0
        self.beta_v0 = beta_v0

        self.mu_v = mu_v0
        self.lambda_v = lambda_v0
        self.alpha_v = alpha_v0
        self.beta_v = beta_v0

        self.n_v = 0
        self.q_v = 0
        self.x2_v = 0

        return self
    end
end

function outputArmModel(am::ArmModel)

    println("N: ", am.N, ", V: ", am.V)

    if am.n > 1
        s = 1 / am.n * am.x2 - am.q^2

        if abs(s) < 1e-7
            s = 0.
        end

    else
        s = 0.

    end

    println("n: ", am.n, ", q: ", neat(am.q), ", s: ", neat(s), ", mu: ", neat(am.mu), ", lambda: ", neat(am.lambda), ", alpha: ", neat(am.alpha), ", beta: ", neat(am.beta_))

    if am.n_v > 1
        s = 1 / am.n_v * am.x2_v - am.q_v^2

        if abs(s) < 1e-7
            s = 0.
        end

    else
        s = 0.

    end

    println("n_v: ", am.n_v, ", q_v: ", neat(am.q_v), ", s: ", neat(s), ", mu_v: ", neat(am.mu_v), ", lambda_v: ", neat(am.lambda_v), ", alpha_v: ", neat(am.alpha_v), ", beta_v: ", neat(am.beta_v))
end

function sampleFromModel(am::ArmModel)

    p = rand(Beta(am.V + am.V_alpha0, am.N - am.V + am.V_beta0))

    mu, tau = rand(NormalGamma(am.mu, am.lambda, am.alpha, am.beta_))
    r = rand(Normal(mu, sqrt(1 / tau)))

    mu_v, tau_v = rand(NormalGamma(am.mu_v, am.lambda_v, am.alpha_v, am.beta_v))
    r_v = rand(Normal(mu_v, sqrt(1 / tau_v)))

    # expected reward
    #return (1 - p) * r + p * r_v

    # expected mean
    return (1 - p) * mu + p * mu_v
end

function updateModel(am::ArmModel, q::Float64)

    am.N += 1

    bV = false

    if q < am.v_bound
        am.V += 1

        bV = true
    end

    if !bV
        am.n += 1
        am.q += (q - am.q) / am.n
        am.x2 += q * q

        if am.n > 1
            s = 1 / am.n * am.x2 - am.q^2

            if abs(s) < 1e-7
                s = 0.
            end

        else
            s = 0.

        end

        am.mu = (am.lambda0 * am.mu0 + am.n * am.q) / (am.lambda0 + am.n)
        am.lambda = am.lambda0 + am.n
        am.alpha = am.alpha0 + am.n / 2
        am.beta_ = am.beta0 + 1 / 2 * (am.n * s + am.lambda0 * am.n * (am.q - am.mu0)^2 / (am.lambda0 + am.n))

    else
        am.n_v += 1
        am.q_v += (q - am.q_v) / am.n_v
        am.x2_v += q * q

        if am.n_v > 1
            s = 1 / am.n_v * am.x2_v - am.q_v^2

            if abs(s) < 1e-7
                s = 0.
            end

        else
            s = 0.

        end

        am.mu_v = (am.lambda_v0 * am.mu_v0 + am.n_v * am.q_v) / (am.lambda_v0 + am.n_v)
        am.lambda_v = am.lambda_v0 + am.n_v
        am.alpha_v = am.alpha_v0 + am.n_v / 2
        am.beta_v = am.beta_v0 + 1 / 2 * (am.n_v * s + am.lambda_v0 * am.n_v * (am.q_v - am.mu_v0)^2 / (am.lambda_v0 + am.n_v))

    end
end


function ThompsonSamplingWithModel(rewards, generateArmModel::Function; n::Int64 = 10000, bPlot::Bool = false, plotfunc::Symbol = :semilogx, debug::Int64 = 0, verbose::Int64 = 1)

    nArms = length(rewards)

    AM = generateArmModel(nArms)

    N_total = 0
    N = zeros(Int64, nArms)
    Q = zeros(nArms)

    N_hist = zeros(Int64, nArms, n)
    Q_hist = zeros(nArms, n)
    Qv_hist = zeros(nArms, n)

    V_hist = zeros(nArms, n)
    mu_hist = zeros(nArms, n)
    sigma_hist = zeros(nArms, n)
    mu_v_hist = zeros(nArms, n)
    sigma_v_hist = zeros(nArms, n)

    theta = zeros(nArms)

    if debug > 0
        println("i: 0")
        println()
        for i = 1:nArms
            outputArmModel(AM[i])
            println()
        end
    end

    for i = 1:n
        if debug > 0
            println("i: ", i)
            println()
        end

        for j = 1:nArms
            theta[j] = sampleFromModel(AM[j])
        end

        k = argmax(theta)

        q = rand(rewards[k])

        updateModel(AM[k], q)

        if debug > 0
            for j = 1:nArms
                outputArmModel(AM[j])
                println()
            end
        end

        N_total += 1
        N[k] += 1
        Q[k] += (q - Q[k]) / N[k]

        for j = 1:nArms
            N_hist[j, i] = N[j]
            Q_hist[j, i] = Q[j]
            Qv_hist[j, i] = theta[j]

            if AM[j].N > 0
                V_hist[j, i] = AM[j].V / AM[j].N
            end
            mu_hist[j, i] = AM[j].mu
            sigma_hist[j, i] = sqrt(AM[j].beta_ / AM[j].alpha)
            mu_v_hist[j, i] = AM[j].mu_v
            sigma_v_hist[j, i] = sqrt(AM[j].beta_v / AM[j].alpha_v)
        end
    end

    U = map(mean, rewards)
    bestArm = argmax(U)
    
    Regret = zeros(n)
    Played = zeros(n)

    for i = 1:n
        Regret[i] = dot((U[bestArm] - U), N_hist[:, i])
        Played[i] = N_hist[bestArm, i] / i
    end
    
    if bPlot
        sleep(0.1)

        figure()
        for a_ = 1:nArms
            eval(plotfunc)(1:n, vec(V_hist[a_, :]))
        end
        ylim([0, 1])
        xlabel("Number of plays")
        ylabel("p")
        legend([string(i) for i = 1:nArms], loc = "best")

        figure()
        for a_ = 1:nArms
            eval(plotfunc)(1:n, vec(mu_hist[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("mu")
        legend([string(i) for i = 1:nArms], loc = "best")
        #legend(vec([string(i) * string(j) for j in ["", "v"], i = 1:nArms]), loc = "best")

        figure()
        for a_ = 1:nArms
            eval(plotfunc)(1:n, vec(sigma_hist[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("sigma")
        legend([string(i) for i = 1:nArms], loc = "best")
        #legend(vec([string(i) * string(j) for j in ["", "v"], i = 1:nArms]), loc = "best")

        figure()
        for a_ = 1:nArms
            eval(plotfunc)(1:n, vec(mu_v_hist[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("mu_v")
        legend([string(i) for i = 1:nArms], loc = "best")

        figure()
        for a_ = 1:nArms
            eval(plotfunc)(1:n, vec(sigma_v_hist[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("sigma_v")
        legend([string(i) for i = 1:nArms], loc = "best")

        if verbose > 0
            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(N_hist[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("N")
            legend([string(i) for i = 1:nArms], loc = "best")
        end

        figure()
        for a_ = 1:nArms
            eval(plotfunc)(1:n, vec(Q_hist[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("Q")
        legend([string(i) for i = 1:nArms], loc = "best")

        if verbose > 0
            figure()
            for a_ = 1:nArms
                #eval(plotfunc)(1:n, vec(Qv_hist[a_, :]))
                plot(1:n, vec(Qv_hist[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("Qv")
            legend([string(i) for i = 1:nArms], loc = "best")
        end

        figure()
        eval(plotfunc)(1:n, Regret)
        xlabel("Number of plays")
        ylabel("Regret")

        figure()
        eval(plotfunc)(1:n, Played * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best arm played")

        println("Best arm: ", bestArm, ", Best arm played: ", neat(Played[end] * 100), "%", ", U: ", neat(U))
        sleep(0.1)
    end
    
    return N_total, N, N_hist, Q, Q_hist, Qv_hist, Regret, Played, V_hist, mu_hist, sigma_hist, mu_v_hist, sigma_v_hist
end


function runExp(rewards, tree_policy; n::Int64 = 10000, bPlot::Bool = false, plotfunc::Symbol = :semilogx, debug::Int64 = 0, verbose::Int64 = 0)

    if tree_policy[1] == :A_UCB
        if length(tree_policy) == 2
            return A_UCB(rewards, tree_policy[2], n = n, bPlot = bPlot, plotfunc = plotfunc)
        elseif length(tree_policy) == 3
            return A_UCB(rewards, tree_policy[2], n = n, policy = tree_policy[3], bPlot = bPlot, plotfunc = plotfunc)
        end

    elseif tree_policy[1] == :TS
        return ThompsonSampling(rewards, n = n, bPlot = bPlot, plotfunc = plotfunc)

    elseif tree_policy[1] == :TS_M
        return ThompsonSamplingWithModel(rewards, tree_policy[2], n = n, bPlot = bPlot, plotfunc = plotfunc)

    else
        nArms = length(rewards)

        Q_hist = zeros(nArms, n)
        Qv_hist = zeros(nArms, n)
        N_hist = zeros(nArms, n)
        Var_hist = zeros(nArms, n)

        N_total = 0
        N = zeros(Int64, nArms)
        Q = zeros(nArms)
        X2 = zeros(nArms)

        if tree_policy[1] == :Exp3
            g = n
            gam = min(1, sqrt(nArms * log(nArms) / ((e - 1) * g)))

            w = ones(nArms)
            p = zeros(nArms)
        end

        for i = 1:n
            Qv = zeros(nArms)
            var_ = zeros(nArms)

            if tree_policy[1] == :Exp3
                sum_w = sum(w)
            end

            for a_ = 1:nArms
                if tree_policy[1] == :Exp3
                    p[a_] = (1 - gam) * w[a_] / sum_w + gam / nArms

                else
                    if N[a_] == 0
                        Qv[a_] = Inf

                    else
                        if N[a_] > 1
                            var_[a_] = (X2[a_] - N[a_] * (Q[a_] * Q[a_])) / (N[a_] - 1)
                            if abs(var_[a_]) < 1.e-7
                                var_[a_] = 0.
                            end
                        end

                        if tree_policy[1] == :UCB1
                            if length(tree_policy) > 1
                                c = tree_policy[2]
                            else
                                c = sqrt(2)
                            end
                            Qv[a_] = Q[a_] + c * sqrt(log(N_total) / N[a_])

                        elseif tree_policy[1] == :UCB1_tuned
                            if length(tree_policy) > 1
                                var_max = tree_policy[2]
                            else
                                var_max  = 1/4
                            end
                            Qv[a_] = Q[a_] + sqrt(log(N_total) / N[a_] * min(var_max, var_[a_] + sqrt(2 * log(N_total) / N[a_])))

                        elseif tree_policy[1] == :UCB_V
                            if length(tree_policy) > 1
                                c = tree_policy[2]
                            else
                                c = 1.
                            end
                            Qv[a_] = Q[a_] + sqrt(2 * var_[a_] * log(N_total) / N[a_]) + c * 3 * log(N_total) / N[a_]

                        elseif tree_policy[1] == :UCB_VAR
                            if length(tree_policy) > 1
                                c1, c2 = tree_policy[2]
                            else
                                c1 = sqrt(1/2)
                                c2 = 1.
                            end
                            # variance of reward
                            #c = c2 * sqrt(var_[a_])
                            # XXX
                            # variance of Q or variance of mean of reward
                            #c = c2 * sqrt(var_[a_] / N[a_])
                            #if c < c1
                            #    c = c1
                            #end
                            c = c1 + c2 * sqrt(var_[a_] / N[a_])
                            Qv[a_] = Q[a_] + c * sqrt(log(N_total) / N[a_])

                        else
                            error(tree_policy[1], " does not exist")

                        end

                    end

                end
            end

            if tree_policy[1] == :Exp3
                a = sampleFromProb(p)
            else
                a = argmax(Qv)
                #a = argmax(Q)
                #a = rand(1:nArms)
            end

            q = rand(rewards[a])

            if tree_policy[1] == :Exp3
                w[a] *= exp(gam * q / p[a] / nArms)
            end

            N_total += 1
            N[a] += 1
            Q[a] += (q - Q[a]) / N[a]
            X2[a] += q * q

            if debug > 1
                println(N_total, ", ", a, ", ", Q, ", ", N)
                if debug > 2
                    println(Qv)
                end
            end

            for a_ = 1:nArms
                Q_hist[a_, i] = Q[a_]
                Qv_hist[a_, i] = Qv[a_]
                N_hist[a_, i] = N[a_]
                Var_hist[a_, i] = var_[a_]
            end
        end

        U = map(mean, rewards)
        bestArm = argmax(U)

        Regret = zeros(n)
        Played = zeros(n)

        for i = 1:n
            Regret[i] = dot((U[bestArm] - U), N_hist[:, i])
            Played[i] = N_hist[bestArm, i] / i
        end

        if bPlot
            if verbose > 0
                figure()
                for a_ = 1:nArms
                    eval(plotfunc)(1:n, vec(N_hist[a_, :]))
                end
                xlabel("Number of plays")
                ylabel("Number of plays of each arm")
                legend([string(i) for i = 1:nArms], loc = "best")
            end

            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(Q_hist[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("Q")
            legend([string(i) for i = 1:nArms], loc = "best")

            if verbose > 0
                figure()
                for a_ = 1:nArms
                    eval(plotfunc)(1:n, vec(Qv_hist[a_, :]))
                end
                xlabel("Number of plays")
                ylabel("Qv")
                legend([string(i) for i = 1:nArms], loc = "best")
            end

            if verbose > 1
                figure()
                for a_ = 1:nArms
                    eval(plotfunc)(1:n, vec(Var_hist[a_, :]))
                end
                xlabel("Number of plays")
                ylabel("Variance")
                legend([string(i) for i = 1:nArms], loc = "best")
            end

            figure()
            eval(plotfunc)(1:n, Regret)
            xlabel("Number of plays")
            ylabel("Regret")

            figure()
            eval(plotfunc)(1:n, Played * 100)
            ylim([0, 100])
            xlabel("Number of plays")
            ylabel("% of best arm played")
        end

        return N_total, N, N_hist, Q, Q_hist, Qv_hist, Regret, Played
    end
end


function runExpN(rewards, tree_policy; n::Int64 = 10000, N::Int64 = 100, bPlot::Bool = false, plotfunc::Symbol = :semilogx, verbose::Int64 = 0)

    nArms = length(rewards)

    N_hist_acc = zeros(nArms, n)
    Q_hist_acc = zeros(nArms, n)
    Qv_hist_acc = zeros(nArms, n)
    Regret_acc = zeros(n)
    Played_acc = zeros(n)
    nBestArm = 0

    N_end = zeros(nArms, N)
    Q_end = zeros(nArms, N)
    Qv_end = zeros(nArms, N)
    Regret_end = zeros(N)
    Played_end = zeros(N)

    if tree_policy[1] == :A_UCB
        c_hist_acc = zeros(length(tree_policy[2]), n)
        Qc_hist_acc = zeros(length(tree_policy[2]), n)
    elseif tree_policy[1] == :TS
        S_hist_acc = zeros(nArms, n)
    elseif tree_policy[1] == :TS_M
        V_hist_acc = zeros(nArms, n)
        mu_hist_acc = zeros(nArms, n)
        sigma_hist_acc = zeros(nArms, n)
        mu_v_hist_acc = zeros(nArms, n)
        sigma_v_hist_acc = zeros(nArms, n)
    end
    
    U = map(mean, rewards)
    bestArm = argmax(U)

    for i = 1:N
        if tree_policy[1] == :A_UCB
            N_total, N_, N_hist, Q, Q_hist, Qv_hist, Regret, Played, c_hist, Qc_hist = runExp(rewards, tree_policy, n = n)
        elseif tree_policy[1] == :TS
            N_total, N_, N_hist, Q, Q_hist, Qv_hist, Regret, Played, S_hist = runExp(rewards, tree_policy, n = n)
        elseif tree_policy[1] == :TS_M
            N_total, N_, N_hist, Q, Q_hist, Qv_hist, Regret, Played, V_hist, mu_hist, sigma_hist, mu_v_hist, sigma_v_hist = runExp(rewards, tree_policy, n = n)
        else
            N_total, N_, N_hist, Q, Q_hist, Qv_hist, Regret, Played = runExp(rewards, tree_policy, n = n)

        end

        N_hist_acc += (N_hist .- N_hist_acc) ./ i
        Q_hist_acc += (Q_hist .- Q_hist_acc) ./ i
        Qv_hist_acc += (Qv_hist .- Qv_hist_acc) ./ i
        Regret_acc += (Regret .- Regret_acc) ./ i
        Played_acc += (Played .- Played_acc) ./ i
        if argmax(Q) == bestArm
            nBestArm += 1
        end

        for j = 1:nArms
            N_end[j, i] = N_hist[j, end]
            Q_end[j, i] = Q_hist[j, end]
            Qv_end[j, i] = Qv_hist[j, end]
        end
        Regret_end[i] = Regret[end]
        Played_end[i] = Played[end]

        if tree_policy[1] == :A_UCB
            c_hist_acc += (c_hist .- c_hist_acc) ./ i
            Qc_hist_acc += (Qc_hist .- Qc_hist_acc) ./ i
        elseif tree_policy[1] == :TS
            S_hist_acc += (S_hist .- S_hist_acc) ./ i
        elseif tree_policy[1] == :TS_M
            V_hist_acc += (V_hist .- V_hist_acc) ./ i
            mu_hist_acc += (mu_hist .- mu_hist_acc) ./ i
            sigma_hist_acc += (sigma_hist .- sigma_hist_acc) ./ i
            mu_v_hist_acc += (mu_v_hist .- mu_v_hist_acc) ./ i
            sigma_v_hist_acc += (sigma_v_hist .- sigma_v_hist_acc) ./ i
        end
    end
    
    if bPlot
        for i = 1:nArms
            println("Arm ", i, ": ", string(rewards[i]))
        end

        print("mean of reward:")
        for i = 1:nArms
            print((i == 1) ? " " : ", ", neat(U[i]))
        end
        println()
        println()

        print("Policy: ")
        if tree_policy[1] == :TS_M
            println(string(tree_policy[1]))
        else
            str = string(tree_policy[1])
            if length(tree_policy) > 1
                str *= " ["
            end
            for i = 2:length(tree_policy)
                if i > 2
                    str *= ", "
                end
                str *= string(tree_policy[i])
            end
            if length(tree_policy) > 1
                str *= "]"
            end
            println(str)
        end
        println()

        println("Best arm: ", bestArm)
        println("Best arm played: ", neat(Played_acc[end] * 100), "%")
        println()

        if verbose > 0
            print("std(N[end]):")
            for i = 1:nArms
                print((i == 1) ? " " : ", ", neat(std(vec(N_end[i, :]))))
            end
            println()
        end

        print("std(Q[end]):")
        for i = 1:nArms
            print((i == 1) ? " " : ", ", neat(std(vec(Q_end[i, :]))))
        end
        println()

        if verbose > 0
            print("std(Qv[end]):")
            for i = 1:nArms
                print((i == 1) ? " " : ", ", neat(std(vec(Qv_end[i, :]))))
            end
            println()
    end

        print("std(Regret[end]): ", neat(std(Regret_end)))
        println()

        print("std(Played[end]): ", neat(std(Played_end * 100)))
        println()

        println()
        sleep(0.1)

        if tree_policy[1] == :A_UCB
            figure()
            for i = 1:length(tree_policy[2])
                eval(plotfunc)(1:n, vec(c_hist_acc[i, :]) * 100)
            end
            ylim([0, 100])
            xlabel("Number of plays")
            ylabel("% of c played")
            legend([string(round(Int64, tree_policy[2][i])) for i = 1:length(tree_policy[2])], loc = "best")

            figure()
            for i = 1:length(tree_policy[2])
                eval(plotfunc)(1:n, vec(Qc_hist_acc[i, :]))
            end
            xlabel("Number of plays")
            ylabel("Qc")
            legend([string(round(Int64, tree_policy[2][i])) for i = 1:length(tree_policy[2])], loc = "best")

        elseif tree_policy[1] == :TS
            figure()
            for i = 1:nArms
                eval(plotfunc)(1:n, vec(S_hist_acc[i, :]))
            end
            ylim([0, 1])
            xlabel("Number of plays")
            ylabel("Theta")
            legend([string(i) for i = 1:nArms], loc = "best")

        elseif tree_policy[1] == :TS_M
            figure()
            for i = 1:nArms
                eval(plotfunc)(1:n, vec(V_hist_acc[i, :]))
            end
            ylim([0, 1])
            xlabel("Number of plays")
            ylabel("p")
            legend([string(i) for i = 1:nArms], loc = "best")

            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(mu_hist_acc[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("mu")
            legend([string(i) for i = 1:nArms], loc = "best")
            #legend(vec([string(i) * string(j) for j in ["", "v"], i = 1:nArms]), loc = "best")

            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(sigma_hist_acc[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("sigma")
            legend([string(i) for i = 1:nArms], loc = "best")
            #legend(vec([string(i) * string(j) for j in ["", "v"], i = 1:nArms]), loc = "best")

            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(mu_v_hist_acc[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("mu_v")
            legend([string(i) for i = 1:nArms], loc = "best")

            figure()
            for a_ = 1:nArms
                eval(plotfunc)(1:n, vec(sigma_v_hist_acc[a_, :]))
            end
            xlabel("Number of plays")
            ylabel("sigma_v")
            legend([string(i) for i = 1:nArms], loc = "best")

        end

        if verbose > 0
            figure()
            for i = 1:nArms
                eval(plotfunc)(1:n, vec(N_hist_acc[i, :]))
            end
            xlabel("Number of plays")
            ylabel("N")
            legend([string(i) for i = 1:nArms], loc = "best")
        end

        figure()
        for i = 1:nArms
            eval(plotfunc)(1:n, vec(Q_hist_acc[i, :]))
        end
        xlabel("Number of plays")
        ylabel("Q")
        legend([string(i) for i = 1:nArms], loc = "best")

        if verbose > 0
            figure()
            for i = 1:nArms
                if tree_policy[1] == :TS_M
                    plot(1:n, vec(Qv_hist_acc[i, :]))
                else
                    eval(plotfunc)(1:n, vec(Qv_hist_acc[i, :]))
                end
            end
            xlabel("Number of plays")
            ylabel("Qv")
            legend([string(i) for i = 1:nArms], loc = "best")
        end
        
        figure()
        eval(plotfunc)(1:n, Regret_acc)
        xlabel("Number of plays")
        ylabel("Regret")

        figure()
        eval(plotfunc)(1:n, Played_acc * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best arm played")
    end
    
    return Regret_acc, Played_acc, nBestArm
end


function plotExpParam(rewards, tree_policy, params; n::Int64 = 10000, N::Int64 = 100, plotfunc::Symbol = :semilogx)

    labels = ASCIIString[]

    for param in params
        push!(labels, string(neat(param)))

        Regret_acc, Played_acc, nBestArm = runExpN(rewards, Any[tree_policy, param], n = n, N = N)

        #println("Tree Policy: ", tree_policy[1], ", Param: ", param, ", nBestArm: ", nBestArm, " / ", N)
        #sleep(0.1)
        
        figure(1)
        eval(plotfunc)(1:n, Regret_acc)
        xlabel("Number of plays")
        ylabel("Regret")
        legend(labels, loc = "best")

        figure(2)
        eval(plotfunc)(1:n, Played_acc * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best arm played")
        legend(labels, loc = "best")
    end
end


function plotExpPolicy(rewards, tree_policies; n::Int64 = 10000, N::Int64 = 100, plotfunc::Symbol = :semilogx, bPlotAvgRegret::Bool = false)

    nArms = length(rewards)

    U = map(mean, rewards)
    bestArm = argmax(U)

    for i = 1:nArms
        println("Arm ", i, ": ", string(rewards[i]))
    end

    print("mean of reward:")
    for i = 1:nArms
        print((i == 1) ? " " : ", ", neat(U[i]))
    end
    println()
    println()

    println("Best arm: ", bestArm)
    println()

    sleep(0.1)

    labels = ASCIIString[]

    for tree_policy in tree_policies
        if tree_policy[1] == :TS_M
            push!(labels, string(tree_policy[1]))
        else
            str = string(tree_policy[1])
            if length(tree_policy) > 1
                str *= " ["
            end
            for i = 2:length(tree_policy)
                if i > 2
                    str *= ", "
                end
                str *= string(tree_policy[i])
            end
            if length(tree_policy) > 1
                str *= "]"
            end
            push!(labels, str)
        end

        Regret_acc, Played_acc, nBestArm = runExpN(rewards, tree_policy, n = n, N = N)

        #println("Tree Policy: ", tree_policy[1], ", nBestArm: ", nBestArm, " / ", N)
        #sleep(0.1)
        
        figure(1)
        if !bPlotAvgRegret
            eval(plotfunc)(1:n, Regret_acc)
        else
            eval(plotfunc)(1:n, Regret_acc ./ collect(1:n))
        end
        xlabel("Number of plays")
        ylabel("Regret")
        legend(labels, loc = "best")

        figure(2)
        eval(plotfunc)(1:n, Played_acc * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best arm played")
        legend(labels, loc = "best")
    end
end


