function init_cluster_sherlock()

    if !haskey(ENV, "SLURM_JOB_NODELIST")
        error("SLURM_JOB_NODELIST not defined")
    end

    if !haskey(ENV, "SLURM_NTASKS_PER_NODE")
        error("SLURM_NTASKS_PER_NODE not defined")
    end

    n = parse(Int64, ENV["SLURM_NTASKS_PER_NODE"])
    dir = "/home/youngjun/MultiArmedBandit"

    machines = Tuple{ASCIIString, Int64}[]

    list = ENV["SLURM_JOB_NODELIST"]

    for m = eachmatch(r"([\w\d-]+)(\[[\d,-]+\])?", list)
        if m.captures[2] == nothing
            push!(machines, (m.captures[1], n))

        else
            host_pre = m.captures[1]

            s = split(m.captures[2][2:end-1], ",")
            for s_ in s
                if isdigit(s_)
                    push!(machines, (host_pre * s_, n))
                else
                    a, b = split(s_, "-")
                    for i = parse(Int64, a):parse(Int64, b)
                        push!(machines, (host_pre * string(i), n))
                    end
                end
            end

        end
    end

    for (machine, count) in machines
        cluster_list = ASCIIString[]

        for i = 1:count
            push!(cluster_list, machine)
        end

        addprocs(cluster_list, dir = dir)
    end
end


if contains(gethostname(), "sherlock")
    init_cluster_sherlock()
else
    addprocs(CPU_CORES - 2)
end


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


param_list = vec([[p1, -1000, -p2, p2 / rand(2:5), p3, -1000, -p4, p4 / rand(2:5)] for p1 = 0:0.01:0.1, p2 = 10:10:100, p3 = 0:0.01:0.1, p4 = 10:10:100])

#results = runRangeTest(rand_param())
results = pmap(runRangeTest_, param_list)

save("rtest.jld", "results", results)


