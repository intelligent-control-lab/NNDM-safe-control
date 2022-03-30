using NeuralVerification
using Ipopt
using JuMP
# include("safe_set.jl")
include("unicycle_env.jl");
mutable struct NvController
    solver
    ϵ
    u_lim
    err_bound
    start_values
    warm_start
    bin_precision
end
function NvController(err_bound; warm_start=true, bin_precision=1e-3)
    NvController(NNDynTrack(), 1e-8, [4., π], err_bound, nothing, warm_start, bin_precision)
end
function NvController()
    NvController(NNDynTrack(), 1e-8, [4., π], [1e9, 1e9, 1e9, 1e9], nothing, false, 1e-3)
end

mutable struct AdamNvController
    solver
    ϵ
    u_lim
    err_bound
    start_values
    warm_start
    bin_precision
    num_sample
    adam_ctrl
end
function AdamNvController(num_sample)
    AdamNvController(NNDynTrack(), 1e-8, [4., π], [1e9, 1e9, 1e9, 1e9], nothing, true, 1e-3,  num_sample, AdamBAController(num_sample))
end

mutable struct NlController
    ϵ
    u_lim
    err_bound
    start_values
    warm_start
    iter
end
function NlController(err_bound; warm_start=true, iter=1)
    NlController(1e-8, [4., π], err_bound, nothing, warm_start, iter)
end

mutable struct ShootingController
    u_lim
    num_sample
end
function ShootingController(num_sample)
    ShootingController([4., π], num_sample)
end

mutable struct AdamBAController
    u_lim
    num_sample
    num_du_sample
    l0
    MIND_init
end
function AdamBAController(num_sample, num_du_sample)
    AdamBAController([4., π], num_sample, num_du_sample, 1e-1, false)
end
function AdamBAController(num_du_sample)
    AdamBAController([4., π], 0, num_du_sample, 1e-1, true)
end

function linear_cost(x, xref, obj_cost)
    sum(abs.(x - xref).*obj_cost)
end
function quad_cost(x, xref, obj_cost)
    sum((x - xref).^2 .* obj_cost)
end

function is_safe_u(net, x, u, safe_set)
    dot_x = compute_output(net, [x;u])
    return dot_x ∈ safe_set
end
function exceed_lim(ctrl, u)
    return any(u .> ctrl.u_lim) || any(u .< -ctrl.u_lim)
end
function find_adam_control(ctrl::AdamBAController, net, x, u0, du, r, safe_set)
    is_safe_u(net, x, u0, safe_set) && return u0

    while true
        u = u0 + du * r
        (is_safe_u(net, x, u, safe_set) || exceed_lim(ctrl, u)) && break
        r = r * 2
    end
    l = 0
    eps = 1e-6
    while (r - l) > eps
        m = (l+r)/2.0
        u = u0 + du * m
        if exceed_lim(ctrl, u) || is_safe_u(net, x, u, safe_set)
            r = m
        else
            l = m
        end
    end
    return u0 + du * r
end


function get_control(ctrl::AdamNvController, xref, x, net, obj_cost, dt; obstacles=nothing, safety_index=nothing, u_ref=nothing)
    input = Hyperrectangle(low=[x.-ctrl.ϵ; -ctrl.u_lim], high=[x.+ctrl.ϵ; ctrl.u_lim])
    safe_set = isnothing(obstacles) ? HalfSpace(zero(x).+1.0,Inf) : phi_safe_set(safety_index, x, obstacles, dt)
    dot_x_ref = (xref-x)/dt
    
    start_values = nothing
    result = nothing

    output = Hyperrectangle(dot_x_ref, ctrl.err_bound/dt)
    isnothing(obstacles) || (output = intersection(output, safe_set))
    problem = TrackingProblem(net, input, output, dot_x_ref, obj_cost* (dt^2))
    
    # adam_ctrl = AdamBAController(ctrl.num_sample)
    u_init, safe_set = get_control(ctrl.adam_ctrl, xref, x, net, obj_cost, dt; obstacles=obstacles, safety_index=safety_index, u_ref=u_ref)
    u_init = isnothing(u_init) ? u_ref : u_init
    result, start_values = NeuralVerification.solve(ctrl.solver, problem, ctrl.start_values, u_ref=u_ref, xu_init=[x; u_init])
    
    result.status == :violated && (return nothing, nothing)
    
    ctrl.warm_start && (ctrl.start_values = start_values)
    u = result.input[5:6]

    return u, safe_set
end

function get_control(ctrl::AdamBAController, xref, x, net, obj_cost, dt; obstacles=nothing, safety_index=nothing, u_ref=nothing)
    min_loss=1e9
    u = nothing
    
    safe_set = isnothing(obstacles) ? HalfSpace(zero(x).+1.0,Inf) : phi_safe_set(safety_index, x, obstacles, dt)

    if ctrl.MIND_init || !isnothing(u_ref)
        if !isnothing(u_ref)
            u0_cand = u_ref
        else
            u0_cand, _ = get_control(NvController(), xref, x, net, obj_cost, dt)
        end
        # is_safe_u(net, x, u0_cand, safe_set) && return u0_cand, safe_set
        for k in 1:ctrl.num_du_sample
            du_cand = rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim
            du_cand = normalize(du_cand)
            u_cand = find_adam_control(ctrl, net, x, u0_cand, du_cand, ctrl.l0, safe_set)
            dot_x_cand = compute_output(net, [x; u_cand])
            dot_x_cand ∈ safe_set || continue
            x_cand = forward(net, x, u_cand, dt)
            if isnothing(u_ref)
                if quad_cost(x_cand, xref, obj_cost) < min_loss
                    min_loss = quad_cost(x_cand, xref, obj_cost)
                    u = u_cand
                end
            else    
                loss = dot(u_cand - u_ref, u_cand - u_ref) 
                if loss < min_loss
                    min_loss = loss
                    u = u_cand
                end
            end
        end
    else 
        for j in 1:ctrl.num_sample
            u0_cand = rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim
            for k in 1:ctrl.num_du_sample
                du_cand = rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim
                du_cand = normalize(du_cand)
                u_cand = find_adam_control(ctrl, net, x, u0_cand, du_cand, ctrl.l0, safe_set)
                dot_x_cand = compute_output(net, [x; u_cand])
                dot_x_cand ∈ safe_set || continue
                x_cand = forward(net, x, u_cand, dt)
                if quad_cost(x_cand, xref, obj_cost) < min_loss
                    min_loss = quad_cost(x_cand, xref, obj_cost)
                    u = u_cand
                end
            end
        end
    end    
    return u, safe_set
end


function get_control(ctrl::ShootingController, xref, x, net, obj_cost, dt; obstacles=nothing, safety_index=nothing, u_ref=nothing)
    min_loss=1e9
    u = nothing
    
    safe_set = isnothing(obstacles) ? HalfSpace(zero(x).+1.0,Inf) : phi_safe_set(safety_index, x, obstacles, dt)
    
    A, b = tosimplehrep(safe_set)
    min_vio = 1e9
    u_most_safe = nothing
    for j in 1:ctrl.num_sample
        u_cand = rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim
        dot_x_cand = compute_output(net, [x; u_cand])
        vio = -1e9
        for i in 1:length(b)
            vio = max(vio, A[i,:]' * dot_x_cand - b[i])
        end
        if vio < min_vio 
            min_vio = vio
            u_most_safe = u_cand
        end
        dot_x_cand ∈ safe_set || continue
        x_cand = forward(net, x, u_cand, dt)
        if isnothing(u_ref)
            if quad_cost(x_cand, xref, obj_cost) < min_loss
                min_loss = quad_cost(x_cand, xref, obj_cost)
                u = u_cand
            end
        else     
            if dot(u_cand - u_ref, u_cand - u_ref) < min_loss
                min_loss = dot(u_cand - u_ref, u_cand - u_ref)
                u = u_cand
            end
        end
    end
    # if isnothing(u)
    #     @show min_vio
    # end
    isnothing(u) && (u = u_most_safe)
    return u, safe_set
end

function get_mpc(ctrl::ShootingController, tp, x, k, h)
    min_loss=1e9
    X = [copy(x) for i = 1:h+1]
    U = [zeros(size(tp.u_lim)) for i = 1:h]
    for j in 1:ctrl.num_sample
        U_cand = [rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim  for i = 1:h]
        X_cand = forward(net, x, U_cand, tp.dt)
        mpc_cost = sum([sum(abs.(X_cand[i] - tp.Xref[min(k+i, tp.T)]).*tp.obj_cost) for i = 1:h])
        if mpc_cost < min_loss
            min_loss = mpc_cost
            U = U_cand
        end
    end
    X[2:h+1] = forward(net, x, U, tp.dt)
    return X, U
end

function get_control(ctrl::NvController, xref, x, net, obj_cost, dt; obstacles=nothing, safety_index=nothing, u_ref=nothing)
    input = Hyperrectangle(low=[x.-ctrl.ϵ; -ctrl.u_lim], high=[x.+ctrl.ϵ; ctrl.u_lim])
    safe_set = isnothing(obstacles) ? HalfSpace(zero(x).+1.0,Inf) : phi_safe_set(safety_index, x, obstacles, dt)
    dot_x_ref = (xref-x)/dt
    
    start_values = nothing
    result = nothing

    output = Hyperrectangle(dot_x_ref, ctrl.err_bound/dt)
    isnothing(obstacles) || (output = intersection(output, safe_set))
    problem = TrackingProblem(net, input, output, dot_x_ref, obj_cost* (dt^2))
    result, start_values = NeuralVerification.solve(ctrl.solver, problem, ctrl.start_values, u_ref=u_ref, xu_init= isnothing(u_ref) ? nothing : [x; u_ref])
    
    result.status == :violated && (return nothing, nothing)
    
    ctrl.warm_start && (ctrl.start_values = start_values)
    u = result.input[5:6]

    return u, safe_set
end

function get_control(ctrl::NlController, xref, x, net, obj_cost, dt; obstacles=nothing, u_ref=nothing)
    lb = [x.-ctrl.ϵ; -ctrl.u_lim]
    ub = [x.+ctrl.ϵ; ctrl.u_lim]
    
    dot_x_ref = (xref-x)/dt
    
    model = Model(with_optimizer(Ipopt.Optimizer))
    set_silent(model)
    
    dot_x_lb = dot_x_ref - ctrl.err_bound/dt
    dot_x_ub = dot_x_ref + ctrl.err_bound/dt
    
    @variable(model, dot_x_lb[i] <= dot_x[i=1:4] <= dot_x_ub[i])
    @variable(model, lb[i] <= xu[i=1:6] <= ub[i])
    
    isnothing(ctrl.start_values) || set_start_value.(all_variables(model), ctrl.start_values)
    
#     @NLconstraint(model, dot_x == compute_output(net, [x; u]))
    f(x1,x2,x3,x4,u1,u2) = compute_output(net, [x1,x2,x3,x4,u1,u2])
    f1(xu...) = f(xu...)[1]
    f2(xu...) = f(xu...)[2]
    f3(xu...) = f(xu...)[3]
    f4(xu...) = f(xu...)[4]
    JuMP.register(model, :f1, 6, f1, autodiff=true)
    JuMP.register(model, :f2, 6, f2, autodiff=true)
    JuMP.register(model, :f3, 6, f3, autodiff=true)
    JuMP.register(model, :f4, 6, f4, autodiff=true)
    @NLconstraint(model, dot_x[1] == f1(xu...))
    @NLconstraint(model, dot_x[2] == f2(xu...))
    @NLconstraint(model, dot_x[3] == f3(xu...))
    @NLconstraint(model, dot_x[4] == f4(xu...))
    
    @NLobjective(model, Min, sum((dot_x[i]-dot_x_ref[i]) * obj_cost[i] for i in 1:length(x)))
    
    last_res = nothing
    for i in 1:ctrl.iter
        JuMP.optimize!(model)
        if i < ctrl.iter
            last_res = value.(all_variables(model))
            set_start_value.(all_variables(model), last_res)
        end
    end

#     @show JuMP.termination_status(model)
#     @show value.(xu)
#     @show value.(dot_x)
#     @show dot_x_lb
#     @show dot_x_ub
    ctrl.warm_start && (ctrl.start_values = value.(all_variables(model)))
    
    return value.(xu)[5:6], nothing
end