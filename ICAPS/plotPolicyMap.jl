using JLD
using PyPlot
using LaTeXStrings


function loadData(datafile::ASCIIString)

    R = load(datafile, "results")

    D = Dict{Tuple{Float64, Float64, Float64, Float64}, Tuple{Float64, Float64, Float64, Float64, Float64, Float64}}()

    for (param, result) in R
        D[(param[1], param[3], param[5], param[7])] = result[2:7]
    end

    return D
end


function postProc(param::Tuple{Float64, Float64, Float64, Float64}, result::Union{Tuple{Float64, Float64, Float64, Float64, Float64, Float64}, Vector{Float64}}; threshold::Float64 = 0.001, bNewIndex::Bool = false)

    ind = indmin(result)

    if ind == 4 || ind == 6
        return ind
    end

    min_ = result[ind]

    thr = abs((param[1] * -1000 + param[2]) - (param[3] * -1000 + param[4])) * 10000 * threshold + 0.01

    if !bNewIndex
        if (result[4] - min_) < thr && (result[6] - min_) < thr
            if result[4] < result[6]
                return 4
            else
                return 6
            end
        elseif (result[4] - min_) < thr
            return 4
        elseif (result[6] - min_) < thr
            return 6
        end

    else
        if (result[4] - min_) < thr || (result[6] - min_) < thr
            return 7
        end

    end

    return ind
end


function plotPolicy(D::Dict{Tuple{Float64, Float64, Float64, Float64}, Tuple{Float64, Float64, Float64, Float64, Float64, Float64}}, p1::Float64, m1::Float64, p2::Float64, m2::Float64; bFixProb::Bool = true, bPostProc::Bool = true, threshold::Float64 = 0.001, ruleout::Union{Vector{Int64}, Void} = nothing, fig = nothing, bDraw::Bool = true, bSave::Bool = false)

    if bFixProb
        M = zeros(Int64, 10, 10)
        # -10:-10:-100
        for i = 1:10
            for j = 1:10
                m1 = -10. * i
                m2 = -10. * j
                result = collect(D[(p1, m1, p2, m2)])
                if ruleout != nothing
                    for k in ruleout
                        result[k] = typemax(Int64)
                    end
                end
                if bPostProc
                    M[i, j] = postProc((p1, m1, p2, m2), result, threshold = threshold)
                else
                    M[i, j] = indmin(result)
                end
            end
        end
    else
        M = zeros(Int64, 11, 11)
        # 0:0.01:0.1
        for i = 1:11
            for j = 1:11
                p1 = 0.01 * (i - 1)
                p2 = 0.01 * (j - 1)
                result = collect(D[(p1, m1, p2, m2)])
                if ruleout != nothing
                    for k in ruleout
                        result[k] = typemax(Int64)
                    end
                end
                if bPostProc
                    M[i, j] = postProc((p1, m1, p2, m2), result, threshold = threshold)
                else
                    M[i, j] = indmin(result)
                end
            end
        end
    end

    if bDraw
        if fig == nothing
            fig = figure(facecolor = "white")
        end
        ax1 = fig[:add_subplot](111)
        if bFixProb
            ax1[:set_xlim](-0.5, 9.5)
            ax1[:set_xticks](collect(0.5:8.5))
            ax1[:set_xticklabels]([])
            if !bSave
                ax1[:set_xticks](collect(0:9), minor = true)
                ax1[:set_xticklabels]([-10 * i for i = 1:10], minor = true)
                ax1[:set_xlabel](L"$\mu_1$")
            end
            ax1[:set_ylim](-0.5, 9.5)
            ax1[:set_yticks](collect(0.5:8.5))
            ax1[:set_yticklabels]([])
            if !bSave
                ax1[:set_yticks](collect(0:9), minor = true)
                ax1[:set_yticklabels]([-10 * i for i = 1:10], minor = true)
                ax1[:set_ylabel](L"$\mu_2$", labelpad = 15, rotation = 0)
            end
        else
            ax1[:set_xlim](-0.5, 10.5)
            ax1[:set_xticks](collect(0.5:9.5))
            ax1[:set_xticklabels]([])
            if !bSave
                ax1[:set_xticks](collect(0:10), minor = true)
                ax1[:set_xticklabels]([0.01 * i for i = 0:10], minor = true)
                ax1[:set_xlabel](L"$p_1$")
            end
            ax1[:set_ylim](-0.5, 10.5)
            ax1[:set_yticks](collect(0.5:9.5))
            ax1[:set_yticklabels]([])
            if !bSave
                ax1[:set_yticks](collect(0:10), minor = true)
                ax1[:set_yticklabels]([0.01 * i for i = 0:10], minor = true)
                ax1[:set_ylabel](L"$p_2$", labelpad = 15, rotation = 0)
            end
        end
        ax1[:set_aspect]("equal")
        ax1[:grid](true)
        if !bSave
            ax1[:set_title]("Best Policy")
        end
        cax = ax1[:imshow](flipdim(rotl90(M), 1), alpha = 0.5, interpolation = "none", vmin = 1, vmax = 6)
        if !bSave
            cbar = fig[:colorbar](cax, ticks = [1, 2, 3, 4, 5, 6], boundaries = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
            cbar[:ax][:set_yticklabels](["UCB1(CE)", "UCB1(10)", "UCB1(350)", "A-UCB(10,350)", "TS", "TSM"])
        end
    end

    if bSave
        savefig("policy.png")
    end

    return rotl90(M)
end


function compStat(D::Dict{Tuple{Float64, Float64, Float64, Float64}, Tuple{Float64, Float64, Float64, Float64, Float64, Float64}}, bPostProc::Bool = true; threshold::Float64 = 0.001, bNewIndex::Bool = false, ruleout::Union{Vector{Int64}, Void} = nothing)

    if !bNewIndex
        S = zeros(Int64, 6)
    else
        S = zeros(Int64, 7)
    end

    for p1 = 0.:0.01:0.1
        for m1 = -10.:-10:-100
            for p2 = 0.:0.01:0.1
                for m2 = -10.:-10:-100
                    result = collect(D[(p1, m1, p2, m2)])

                    if ruleout != nothing
                        for i in ruleout
                            result[i] = typemax(Int64)
                        end
                    end

                    if bPostProc
                        ind = postProc((p1, m1, p2, m2), result, threshold = threshold, bNewIndex = bNewIndex)
                    else
                        ind = indmin(result)
                    end

                    S[ind] += 1
                end
            end
        end
    end

    return round(S / sum(S), 2)
end


