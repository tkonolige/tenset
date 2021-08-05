import JSON
using DataFrames
import CSV
using Glob
using Statistics

function pmap(f, c)
    out = Vector{typeof(f(c[1]))}(undef, length(c))
    Threads.@threads for i in 1:length(c)
        out[i] = f(c[i])
    end
    out
end

function main()
    df = pmap(glob("$(ARGS[1])/*.json")) do f
        map(readlines(f)) do l
            o = JSON.parse(l)
            config = o["i"]
            input = JSON.parse(config[1][1])
            task = input[1]
            target = config[1][2]
            cost = mean(o["r"][1])
            schedule = JSON.json(config[2][2])
            loop_shape = mapreduce(x -> x[1], *, config[2][2])
            DataFrame(input=JSON.json(input),task=task,target=target,cost=cost,schedule=schedule,loop_shape=loop_shape)
        end |> x -> vcat(x...)
    end |> x -> vcat(x...)

    CSV.write(ARGS[2], df)
end

main()
