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

function linear_cost(x, xref, obj_cost)
    sum(abs.(x - xref).*obj_cost)
end
function quad_cost(x, xref, obj_cost)
    sum((x - xref).^2 .* obj_cost)
end

function get_control(ctrl::ShootingController, xref, x, net, obj_cost, dt; obstacles=nothing, safety_index=nothing)
    min_loss=1e9
    u = nothing
    
    safe_set = isnothing(obstacles) ? HalfSpace(zero(x).+1.0,Inf) : phi_safe_set(safety_index, x, obstacles, dt)
    
    for j in 1:ctrl.num_sample
        u_cand = rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim
        dot_x_cand = compute_output(net, [x; u_cand])
        dot_x_cand ∈ safe_set || continue
        x_cand = forward(net, x, u_cand, dt)
#         @show u_cand
#         @show x_cand - xref
#         @show sum(abs.(x_cand - xref).*obj_cost)
        if linear_cost(x_cand, xref, obj_cost) < min_loss
            min_loss = linear_cost(x_cand, xref, obj_cost)
            u = u_cand
        end
    end
    
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

function get_control(ctrl::NvController, xref, x, net, obj_cost, dt; obstacles=nothing, safety_index=nothing)
    input = Hyperrectangle(low=[x.-ctrl.ϵ; -ctrl.u_lim], high=[x.+ctrl.ϵ; ctrl.u_lim])
    safe_set = isnothing(obstacles) ? HalfSpace(zero(x).+1.0,Inf) : phi_safe_set(safety_index, x, obstacles, dt)
    dot_x_ref = (xref-x)/dt
    
    start_values = nothing
    result = nothing
    
    l = 0
    r = 1
    while (r-l) > ctrl.bin_precision
        m = (l+r)/2
        err_bound = ctrl.err_bound/dt * m
        output = Hyperrectangle(dot_x_ref, err_bound)
        isnothing(obstacles) || (output = intersection(output, safe_set))
        problem = TrackingProblem(net, input, output, dot_x_ref, obj_cost)
        result, start_values = NeuralVerification.solve(ctrl.solver, problem, ctrl.start_values)
        result.status == :holds ? (r=m) : (l=m)
    end
    
    output = Hyperrectangle(dot_x_ref, ctrl.err_bound/dt * r)
    isnothing(obstacles) || (output = intersection(output, safe_set))
    problem = TrackingProblem(net, input, output, dot_x_ref, obj_cost)
    result, start_values = NeuralVerification.solve(ctrl.solver, problem, ctrl.start_values)
    
#     while true
#         dot_x_ref = (xref-x)/dt
#         output = Hyperrectangle(dot_x_ref, err_bound)
#         isnothing(obstacles) || (output = intersection(output, safe_set))
#         problem = TrackingProblem(net, input, output, dot_x_ref, obj_cost)
#         result, start_values = NeuralVerification.solve(ctrl.solver, problem, ctrl.start_values)
#         result.status == :holds && break
#         err_bound = err_bound * 1.5
#     end
    
    result.status == :violated && (return nothing, nothing)
    
    ctrl.warm_start && (ctrl.start_values = start_values)
    u = result.input[5:6]

    return u, safe_set
end

function get_control(ctrl::NlController, xref, x, net, obj_cost, dt; obstacles=nothing)
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