
##########  INTEGRATION  ###########

const ∞ = 10.0
# const ∞ = 15.0
const dx = 0.5 #0.005
#const nint = 40

interval = map(x -> sign(x) * abs(x)^2, -1:dx:1) .* ∞
interval_lim(k) = [interval[interval.<k]; k]
interval_lim₊(k) = [k; interval[interval.>k]]
interval⁻ = map(x -> sign(x) * abs(x)^2, -1:dx:0) .* ∞

∫d(f, int=interval) = quadgk(f, int..., atol=1e-7, maxevals=10^7)[1]

∫D(f, int=interval; atol=1e-7) = quadgk(z -> begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., atol=atol, maxevals=10^7)[1]

function ∫D⁻(z0, f)
    z0 < -∞ && return 0.0
    ∫D(f, interval_lim(z0), atol=1e-7)
end

function ∫d⁺(z0, f)
    z0 > ∞ && return 0.0
    ∫d(f, interval_lim₊(z0))
end

function ∫D⁺(z0, f)
    z0 > ∞ && return 0.0
    ∫D(f, interval_lim₊(z0), atol=1e-7)
end



#############################################################


function allheadersshow(io::IO, x...; i0::Int=0)
    i0 == 0 && print(io, "#")
    for y in x
        i0 = headershow(io, y, i0)
        print(io, " ")
    end
    println(io)
end

function recursiveeq(x::T, y::T) where {T}
    all(f -> (getfield(x, f) == getfield(y, f)), fieldnames(T))
end




