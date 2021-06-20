struct RP{P}
    net::P
    obj_cost::Vector{Float64}
    x0::Vector{Float64}
    xg::Vector{Float64}
    u_lim::Vector{Float64}
end
function RP(net, obj_cost, x0, xg; u_lim=[2,4])
    RP(net, float.(obj_cost), float.(x0), float.(xg), float.(u_lim))
end

struct TP{P,Q}
    net::P
    obj_cost::Vector{Float64}
    T::Int
    tf::Float64
    dt::Float64
    x0::Vector{Float64}
    Xref::Vector{Vector{Q}}
    times::Vector{Float64}
    u_lim::Vector{Float64}
end
function TP(net, obj_cost, tf, x0, Xref; u_lim=[2,4])
    T = length(Xref)
    TP(net, float.(obj_cost), T, float(tf), float(tf)/T, x0, Xref, Vector(range(0, tf, length=T)), float.(u_lim))
end
function TP(net, obj_cost, tf, Xref; u_lim=[2,4])
    T = length(Xref)
    TP(net, float.(obj_cost), T, float(tf), float(tf)/T, Xref[1], Xref, Vector(range(0, tf, length=T)), float.(u_lim))
end
function costs(tp, X)
    return [sum(abs.(X[i] - tp.Xref[i]).*tp.obj_cost) for i = 1:tp.T]
end