using Distributions
using PyPlot

import Base.rand


function argmax(A)
    
    max_ = maximum(A)
    indexes = find(A .== max_)
    index = indexes[rand(1:length(indexes))]

    return index
end


type RareDist
    
    p::Float64
    r::Float64
    D::ContinuousUnivariateDistribution

    function RareDist(p::Float64, r::Float64, D::ContinuousUnivariateDistribution)
        
        self = new()
        
        self.p = p
        self.r = r
        self.D = D
        
        return self
    end
end


function rand(D::RareDist)
    
    u = rand()
    
    if u < D.p
        return D.r
    else
        return rand(D.D)
    end
end


function runExp(Rewards, bestArm::Int64, tree_policy, n::Int64 = 100000; bPlot::Bool = false, debug::Int64 = 0)

    nArms = length(Rewards)

    Q_array = zeros(nArms, n)
    N_array = zeros(nArms, n)
    Var_array = zeros(nArms, n)

    N_total = 0
    N = zeros(Int64, nArms)
    Q = zeros(nArms)
    X2 = zeros(nArms)

    for i = 1:n
        Qv = zeros(nArms)
        var_ = zeros(nArms)

        for a_ = 1:nArms
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
                        c1 = 1/4
                        c2 = 1.
                    end
                    c = c2 * sqrt(var_[a_])
                    if c < c1
                        c = c1
                    end
                    Qv[a_] = Q[a_] + c * sqrt(log(N_total) / N[a_])
                else
                    error(tree_policy[1], " does not exist")
                end
            end
        end

        a = argmax(Qv)
        #a = argmax(Q)
        #a = rand(1:nArms)

        q = rand(Rewards[a])

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
            Q_array[a_, i] = Q[a_]
            N_array[a_, i] = N[a_]
            Var_array[a_, i] = var_[a_]
        end
    end

    Regret = zeros(n)
    Played = zeros(n)
    for i = 1:n
        Regret[i] = maximum(Q_array[:, i]) * i - dot(Q_array[:, i], N_array[:, i])
        Played[i] = N_array[bestArm, i] / sum(N_array[:, i])
    end

    if bPlot
        for a_ = 1:nArms
            semilogx(1:n, vec(Q_array[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("Q")
        legend([string(i) for i = 1:nArms], loc = "best")

        figure()
        for a_ = 1:nArms
            semilogx(1:n, vec(N_array[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("Number of plays of each machine")
        legend([string(i) for i = 1:nArms], loc = "best")

        figure()
        for a_ = 1:nArms
            semilogx(1:n, vec(Var_array[a_, :]))
        end
        xlabel("Number of plays")
        ylabel("Variance")
        legend([string(i) for i = 1:nArms], loc = "best")

        figure()
        semilogx(1:n, Regret)
        xlabel("Number of plays")
        ylabel("Regret")

        figure()
        semilogx(1:n, Played * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best machine played")
    end
    
    return N_total, N, Q, Regret, Played
end


function runExpN(N_test::Int64, Rewards, bestArm::Int64, tree_policy, n::Int64 = 100000; bPlot::Bool = false)

    Regret_acc = zeros(n)
    Played_acc = zeros(n)
    nBestArm = 0
    
    for i = 1:N_test
        N_total, N, Q, Regret, Played = runExp(Rewards, bestArm, tree_policy, n)
        Regret_acc += (Regret .- Regret_acc) ./ i
        Played_acc += (Played .- Played_acc) ./ i
        if argmax(Q) == bestArm
            nBestArm += 1
        end
    end
    
    if bPlot
        #println("nBestArm: ", nBestArm, " / ", N_test)
        #sleep(0.1)
        
        semilogx(1:n, Regret_acc)
        xlabel("Number of plays")
        ylabel("Regret")

        figure()
        semilogx(1:n, Played_acc * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best machine played")
    end
    
    return Regret_acc, Played_acc, nBestArm
end


function plotExpParam(Rewards, bestArm::Int64, tree_policy, params; N::Int64 = 100, n::Int64 = 100000)

    labels = ASCIIString[]

    for param in params
        push!(labels, string(param))

        Regret_acc, Played_acc, nBestArm = runExpN(N, Rewards, bestArm, Any[tree_policy, param], n)

        #println("Tree Policy: ", tree_policy[1], ", Param: ", param, ", nBestArm: ", nBestArm, " / ", N)
        #sleep(0.1)
        
        figure(1)
        semilogx(1:n, Regret_acc)
        xlabel("Number of plays")
        ylabel("Regret")
        legend(labels, loc = "best")

        figure(2)
        semilogx(1:n, Played_acc * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best machine played")
        legend(labels, loc = "best")
    end
end


function plotExpPolicy(Rewards, bestArm::Int64, tree_policies; N::Int64 = 100, n::Int64 = 100000)

    labels = ASCIIString[]

    for tree_policy in tree_policies
        push!(labels, string(tree_policy[1]))

        Regret_acc, Played_acc, nBestArm = runExpN(N, Rewards, bestArm, tree_policy, n)

        #println("Tree Policy: ", tree_policy[1], ", nBestArm: ", nBestArm, " / ", N)
        #sleep(0.1)
        
        figure(1)
        semilogx(1:n, Regret_acc)
        xlabel("Number of plays")
        ylabel("Regret")
        legend(labels, loc = "best")

        figure(2)
        semilogx(1:n, Played_acc * 100)
        ylim([0, 100])
        xlabel("Number of plays")
        ylabel("% of best machine played")
        legend(labels, loc = "best")
    end
end


