addprocs(round(Int64, CPU_CORES / 2))

push!(LOAD_PATH, ".")
using coefMod


function rand_param()

    param = zeros(8)

    param[1] = rand() / 10
    param[2] = -1000
    param[3] = -rand(10:100)
    param[4] = abs(param[3] / rand(2:5))

    param[5] = rand() / 10
    param[6] = -1000
    param[7] = -rand(10:100)
    param[8] = abs(param[3] / rand(2:5))

    return param
end


srand(12)

c_min = typemax(Int32)
c_max = typemin(Int32)

for i = 1:10
    param_list = Any[rand_param() for i = 1:4]

    results = pmap(CEOpt_, param_list)

    if c_min > minimum(results)
        c_min = minimum(results)
    end

    if c_max < maximum(results)
        c_max = maximum(results)
    end
    
    println(4 * i, ": ", round(Int32, c_min), " ", round(Int32, c_max))
end


#include("runexp.jl")
#include("ceopt.jl")
#
#rewards = [RareDist(0.1, -1000., Truncated(Normal(-30, 4), -50, -10)), RareDist(0.1, -1000., Truncated(Normal(-70, 10), -120, -20))]
#
#N = 100
#rho = log(N) / N
#
#p = CEOpt(drawSample, [150, 150], computePerf(rewards, 10000, 1), updateParam, N, rho, debug = 1)
#
#println(p)


