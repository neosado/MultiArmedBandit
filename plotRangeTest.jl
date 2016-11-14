using JLD
using PyPlot


function loadData(datafile::ASCIIString)

    R = load(datafile, "results")

    D = Dict{Tuple{Float64, Float64, Float64, Float64}, Tuple{Float64, Float64, Float64, Float64, Float64, Float64}}()

    for (param, result) in R
        # UCB1(CE), UCB1(1), UCB1(240), A-UCB(1,240), TS, TSM
        D[(param[1], param[3], param[5], param[7])] = result[2:7]
    end

    return D
end


function plotPolicy(D::Dict{Tuple{Float64, Float64, Float64, Float64}, Tuple{Float64, Float64, Float64, Float64, Float64, Float64}}, p1::Float64, m1::Float64, p2::Float64, m2::Float64; bFixProb::Bool = true, fig = nothing)

    if bFixProb
        M = zeros(Int64, 10, 10)
        # -10:-10:-100
        for i = 1:10
            for j = 1:10
                m1 = -10. * i
                m2 = -10. * j
                M[i, j] = indmin(D[(p1, m1, p2, m2)])
            end
        end
    else
        M = zeros(Int64, 11, 11)
        # 0:0.01:0.1
        for i = 1:11
            for j = 1:11
                p1 = 0.01 * (i - 1)
                p2 = 0.01 * (j - 1)
                M[i, j] = indmin(D[(p1, m1, p2, m2)])
            end
        end
    end

    if fig == nothing
        fig = figure(facecolor = "white")
    end
    ax1 = fig[:add_subplot](111)
    if bFixProb
        ax1[:set_xlim](-0.5, 9.5)
        ax1[:set_ylim](-0.5, 9.5)
    else
        ax1[:set_xlim](-0.5, 10.5)
        ax1[:set_ylim](-0.5, 10.5)
    end
    ax1[:set_aspect]("equal")
    ax1[:invert_yaxis]()
    if bFixProb
        ax1[:set_xticks](collect(0.5:8.5))
        ax1[:set_yticks](collect(0.5:8.5))
    else
        ax1[:set_xticks](collect(0.5:9.5))
        ax1[:set_yticks](collect(0.5:9.5))
    end
    ax1[:set_xticklabels]([])
    ax1[:set_yticklabels]([])
    ax1[:grid](true)
    ax1[:set_title]("Policy")
    ax1[:imshow](M, alpha = 0.5, interpolation = "none", vmin = 1, vmax = 6)

    return M
end


