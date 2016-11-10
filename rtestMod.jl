module rtestMod

export runRangeTest, runRangeTest_


using coefMod

include("runexp.jl")


function generateArmRewardModel(nArms::Int64)

    AM = Array(ArmModel, nArms)

    for i = 1:nArms
        AM[i] = ArmModel(0.01, 0.01, -50., 1., 1 / 2, 1 / (2 * (1 / 5. ^ 2)), -500., -1000., 1., 1 / 2, 1 / (2 * (1 / 1.^2)))
    end

    return AM
end


function runRangeTest(param)

    n = 10000
    N = 100

    rewards = [RareDist(param[1], param[2], Truncated(Normal(param[3], param[4]), param[3] - 5 * param[4], min(param[3] + 5 * param[4], 0))), RareDist(param[5], param[6], Truncated(Normal(param[7], param[8]), param[7] - 5 * param[8], min(param[7] + 5 * param[8], 0)))]

    c_opt = CEOpt_(param)
    r1, _, _ = runExpN(rewards, Any[:UCB1, c_opt], n = n, N = N)
    r2, _, _ = runExpN(rewards, Any[:UCB1, 1], n = n, N = N)
    r3, _, _ = runExpN(rewards, Any[:UCB1, 240], n = n, N = N)
    r4, _, _ = runExpN(rewards, Any[:AUCB, [genUCBSubArm(1), genUCBSubArm(240)]], n = n, N = N)
    r5, _, _ = runExpN(rewards, Any[:TS], n = n, N = N)
    r6, _, _ = runExpN(rewards, Any[:TSM, generateArmRewardModel], n = n, N = N)

    return c_opt, r1[end], r2[end], r3[end], r4[end], r5[end], r6[end]
end

function runRangeTest_(param)

    return param, runRangeTest(param)
end

end


