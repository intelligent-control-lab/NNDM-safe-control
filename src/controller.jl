mutable struct NvController
    solver
    ϵ
    u_lim
    dx_bound
    start_values
    warm_start
end
function NvController(dx_bound; warm_start=true)
    NvController(NNDynTrack(), 1e-8, [2, 4], dx_bound, nothing, warm_start)
end

mutable struct ShootingController
    u_lim
    num_sample
end
function ShootingController(num_sample)
    ShootingController([2, 4], num_sample)
end


function get_control(ctrl::ShootingController, xref, x, net, obj_cost, dt; obstacle=nothing)
    min_loss=1e9
    u = zeros(2)
    
    safe_set = isnothing(obstacle) ? HalfSpace(zero(x).+1.0,Inf) : phi_safe_set(x, obstacle, dt)
    
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


function get_control(ctrl::NvController, xref, x, net, obj_cost, dt; obstacle=nothing)
    # TODO: implement the controller
    input = Hyperrectangle(low=[x.-ctrl.ϵ; -ctrl.u_lim], high=[x.+ctrl.ϵ; ctrl.u_lim])
    
    dot_x_ref = (xref-x)/dt
    dot_x_bound = ctrl.dx_bound/dt
#     output = Hyperrectangle(dot_x_ref, dot_x_bound)
    output = Hyperrectangle(zero(dot_x_ref), dot_x_bound)
#     safe_set = isnothing(obstacle) ? HalfSpace([0.0],0.0) : phi_safe_set(x, obstacle, dt)
    safe_set = isnothing(obstacle) ? HalfSpace(zero(x).+1.0,Inf) : phi_safe_set(x, obstacle, dt)
    isnothing(obstacle) || (output = intersection(output, safe_set))
    
    problem = TrackingProblem(net, input, output, dot_x_ref, obj_cost)
    result, start_values = solve(ctrl.solver, problem, ctrl.start_values)
    ctrl.warm_start && (ctrl.start_values = start_values)
    u = result.input[5:6]
    
    # C dot_x < d
    # phi(x_k) > 0 -> con: dot_phi(x_k) < -k*phi(x): C = self.grad_phi(x, o)  d = -phi/dt*self.coe[0]
    # phi(x_k) < 0 -> con: phi(x_k+1) < 0:           C = self.grad_phi(x, o)  d = -phi/dt
    
    C = grad_phi(x, obs)
#     @show C
    d = p < 0 ? -p/dt : -p/dt*4
    
    return u, safe_set
end
