function forward(net, x, u::Vector{Float64}, dt)
    dot_x = compute_output(net, [x;u])
    x = x + dot_x * dt
    return x
end

function forward(net, x, U::Vector{Vector{Float64}}, dt)
    h = length(U)
    X = [zeros(size(x)) for k = 1:h];
    for k = 1:h
        x = forward(net, x, U[k], dt)
        X[k] = x
    end
    return X
end

function get_unicycle_endpoints(x)
    theta = x[4]
    ang = (-x[4] + π/2) / π * 180
    s = x[[1,2]]
    t = x[[1,2]] - [cos(theta); sin(theta)]
#     c = s
#     s = s - (t-s)
    return [s[1], t[1]], [s[2], t[2]]
end

function simulate(tp, U, dt; render=false)
    x0 = tp.x0
    net = tp.net

    T = length(U)+1
    X = [zeros(size(x0)) for k = 1:T];
    x = x0
    X[1] = x0
    X[2:T] = forward(net, x0, U::Vector{Vector{Float64}}, dt)
    
    render && visualize(X)
    return X
end

function visualize(X; Xref=nothing, xlims = (-1,4), ylims = (-1,4), obstacle=nothing, safe_sets=nothing)
    step = length(X)
    if !isnothing(Xref)
        xrefs = [Xref[i][1] for i in 1:length(Xref)]
        yrefs = [Xref[i][2] for i in 1:length(Xref)]
    end
    xs = [X[i][1] for i in 1:length(X)]
    ys = [X[i][2] for i in 1:length(X)]

    for i = 1:step
        IJulia.clear_output(true)
        x = X[i]
        sx, sy = get_unicycle_endpoints(x)
        p = plot(sx, sy, xlims = xlims, ylims = ylims, label="", aspect_ratio=:equal)
        scatter!(p, sx[1:1], sy[1:1], label="")
        if !isnothing(Xref)
            plot!(xrefs, yrefs, label="Xref")
            scatter!([xrefs[i]], [yrefs[i]], label="current ref")
        end
        plot!(xs, ys, label="X")
        if !isnothing(obstacle)
            plot!(obstacle, label="obstacle")
        end
        if !isnothing(safe_sets)
            plot!(safe_sets[i], label="obstacle")
        end
        Plots.display(p)
        sleep(0.1)
    end
end