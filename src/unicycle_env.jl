using Plots

function forward(net::Network, x, u::Vector{Float64}, dt)
    dot_x = compute_output(net, [x;u])
    x = x + dot_x * dt
    return x
end


# function forward(net, x, u::Vector{Float64}, dt)
#     dot_x = traj_rk4(x, u, dt)
#     x = x + dot_x * dt
#     return x
# end

function forward(net::Network, x, U::Vector{Vector{Float64}}, dt)
    h = length(U)
    X = [zeros(size(x)) for k = 1:h];
    for k = 1:h
        x = forward(net, x, U[k], dt)
        X[k] = x
    end
    return X
end



function forward(x, u::Vector{Float64}, dt)
    dot_x = traj_rk4(x, u, dt)
    x = x + dot_x * dt
    return x
end

function forward(x, U::Vector{Vector{Float64}}, dt)
    h = length(U)
    X = [zeros(size(x)) for k = 1:h];
    for k = 1:h
        x = forward(gt, x, U[k], dt)
        X[k] = x
    end
    return X
end

function get_unicycle_endpoints(x, l)
    theta = x[4]
    ang = (-x[4] + π/2) / π * 180
    l = 0.4
    s = x[[1,2]]
    t = x[[1,2]] - [cos(theta); sin(theta)] * l
    c = s
    s = s - (t-s)
    wfs = s+(c-s)/3
    wft = s+(c-s)/3*2
    wbs = t-(t-c)/3*2
    wbt = t-(t-c)/3
    return [s[1], t[1]], [s[2], t[2]], [wfs[1], wft[1]], [wfs[2], wft[2]], [wbs[1], wbt[1]], [wbs[2], wbt[2]]
end

function visualize(X; Xref=nothing, Xmpcs=nothing, xlims = nothing, ylims = nothing, obstacles=nothing, targets=nothing, safe_sets=nothing, save_name=nothing, fps=10, save_frame=nothing, traj_label=nothing)
    step = length(X)
    if !isnothing(Xref)
        xrefs = [Xref[i][1] for i in 1:length(Xref)]
        yrefs = [Xref[i][2] for i in 1:length(Xref)]
    end
    xs = [X[i][1] for i in 1:length(X)]
    ys = [X[i][2] for i in 1:length(X)]
    
    xlims == nothing && (xlims = [min(xs), max(xs)])
    ylims == nothing && (ylims = [min(ys), max(ys)])
    anim = @animate for i = 1:step-1
        dpi = isnothing(save_name) & (isnothing(save_frame) || save_frame[1] != i) ? 100 : 300
        IJulia.clear_output(true)
        x = X[i]
        
        l = 0.2
        vx, vy, w1x, w1y, w2x, w2y = get_unicycle_endpoints(x, 0.2)
        p = plot(xtickfontsize=14,ytickfontsize=14,xguidefontsize=14,yguidefontsize=14,legendfontsize=14)
        plot!(p, w1x, w1y, linewidth=43, color=:black, label="")
        plot!(p, w2x, w2y, linewidth=43, color=:black, label="")
        plot!(p, vx, vy, linewidth=34, xlims = xlims, ylims = ylims, color=2, label="", aspect_ratio=:equal, dpi=dpi, legend=:bottomright)

#         p = plot_vehicle(x)
#         p = plot(sx, sy, linewidth=2, color=2, label="", aspect_ratio=:equal, dpi=dpi)
        if !isnothing(obstacles)
            for obs in obstacles
                plot!(p, Ball2(Float64.(obs.center), obs.radius))
            end
        end
        if !isnothing(targets)
            plot!(p, Ball2(Float64.(targets[i].center), targets[i].radius))
        end
#         scatter!(p, c[1:1], c[2:2], label="X center", color=2)
        if !isnothing(Xref)
            plot!(xrefs[1:i], yrefs[1:i], label="Reference Trajectory")
#             scatter!([xrefs[i]], [yrefs[i]], label="current ref")
        end
        if !isnothing(Xmpcs)
            plot!([Xmpcs[i][j][1] for j in 1:length(Xmpcs[i])], [Xmpcs[i][j][2] for j in 1:length(Xmpcs[i])], label="Xmpc")
            scatter!([Xmpcs[i][1][1]], [Xmpcs[i][1][2]], label="current mpc")
        end
        plot!(xs[1:i], ys[1:i], label=isnothing(traj_label) ? "Executed Trajectory" : traj_label, color=3)
        if !isnothing(safe_sets)
            if safe_sets[i] isa HalfSpace
                plot!(HalfSpace(safe_sets[i].a[1:2], safe_sets[i].b), label="safe set")
            else
                safe_set = reduce(intersection, [HalfSpace(con.a[1:2], con.b) for con in safe_sets[i].constraints])
                plot!(safe_set, label="safe set")
            end
        end
        if !isnothing(save_frame) && save_frame[1] == i
            savefig(save_frame[2])
        end
        if isnothing(save_name)
            Plots.display(p)
            sleep(1.0/fps)
        end
    end
    isnothing(save_name) || return gif(anim, save_name, fps = fps)
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

function traj_dynamics(s, u, dt)
    x = s[1]
    y = s[2]
    v = s[3]
    theta = s[4]

    dot_x = v * cos(theta)
    dot_y = v * sin(theta)
    dot_v = u[1]
    dot_theta = u[2]

    dot_s = [dot_x, dot_y, dot_v, dot_theta]
    return dot_s
end

function traj_rk4(s, u, dt)
    dot_s1 = traj_dynamics(s, u, dt)
    dot_s2 = traj_dynamics(s + 0.5*dt*dot_s1, u, dt)
    dot_s3 = traj_dynamics(s + 0.5*dt*dot_s2, u, dt)
    dot_s4 = traj_dynamics(s + dt*dot_s3, u, dt)
    dot_s = (dot_s1 + 2*dot_s2 + 2*dot_s3 + dot_s4)/6.0
    return dot_s
end

function generate_Xref(net, x0, w1, w2, t1, tf, T)
    a = [cos(k/(T-1)*π) for k = 0:T-1]
    T1 = Int(floor(t1/tf * T))
    w = [[w1 for k in 1:T1]; [w2 for k in T1+1:T]]
    Xref = [x0 for k in 1:T]
    Uref = [zeros(2) for k in 1:T-1]
    dt = tf/T
    for k in 1:T-1
        u = [a[k], w[k]]
        Xref[k][3] >  2 && (u[1] = min(u[1], 0))
        Xref[k][3] < -2 && (u[1] = max(u[1], 0))
        Uref[k] = u
        dotx = traj_rk4(Xref[k], u, dt)
#         Xref[k+1] = Xref[k] + dotx * dt
        Xref[k+1] = forward(net, Xref[k], u, dt)
    end
    return Xref, Uref
end

# Xref = generate_Xref([3,1,1,0.2], 1, -2, 1.0, 3.0, 30)

function generate_random_traj(net, num, tf, T)
    Xrefs = []
    Urefs = []
    for i = 1:num
        r = rand(7) .* [2, 2, 2, 2*π, π, π, tf] .- [1, 1, 1, π, π/2, π/2, 0]
        Xref, Uref = generate_Xref(net, r[1:4], r[5], r[6], r[7], tf, T)
        push!(Xrefs, Xref)
        push!(Urefs, Uref)
    end
    return Xrefs, Urefs
end
