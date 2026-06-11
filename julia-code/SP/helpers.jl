
using DataFrames, CSV

function file_to_df(resfile; sort="false", delim=" ")
    df = CSV.read(resfile, DataFrame, delim=delim, ignorerepeated=true)
    df = rename(df, [c => c[findfirst("=", c)[1]+1:end] for c in names(df)])

    if sort != "false"
        sort!(df, [Symbol(sort)])
    end
    return df
end

function df_to_file(df, resfile)
    chars = "#"
    for (i, n) in enumerate(names(df))
        chars *= "$i=$n "
    end
    open(resfile, "w") do rf
        println(rf, chars)
    end
    CSV.write(resfile, eachrow(df), delim=" ", append=true)
end

function combine_stats(df, by, obs_names; avg_abs=false)
    gdf = groupby(df, by)

    dfc = combine(gdf) do group
        len = nrow(group)

        values(n) = avg_abs ? abs.(group[!, n]) : group[!, n]

        means = [Symbol(n) => mean(values(n)) for n in obs_names]
        errs = [Symbol(n, "_err") => std(values(n), corrected=true) / √len for n in obs_names]

        means_and_errs = vcat(reshape(means, 1, :), reshape(errs, 1, :)) |> vec
        (; means_and_errs...)
    end

    return dfc
end

"""
updates the original df with a new row and saves it in the file at path
"""
function save_row!(row, df, path)
    push!(df, row)
    if !isfile(path)
        chars = "#"
        for (i, n) in enumerate(names(df))
            chars *= "$i=$n "
        end
        open(path, "w") do rf
            println(rf, chars)
        end
    end
    CSV.write(path, [row], delim=" ", append=true)
end