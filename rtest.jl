addprocs(CPU_CORES - 2)

using JLD

push!(LOAD_PATH, ".")
using rtestMod


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


#param = rand_param()
#results = runRangeTest(param)


param_list = vec([[p1, -1000, p2, abs(p2 / rand(2:5)), p3, -1000, p4, abs(p4 / rand(2:5))] for p1 = 0:0.1:1, p2 = 10:10:100, p3 = 0:0.1:1, p4 = 10:10:100])

results = pmap(runRangeTest_, param_list)

save("rtest.jld", "result", R)


