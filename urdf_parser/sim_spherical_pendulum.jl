""" 2m spherical pendulum with anchor at origin """
using Quaternions
using StaticArrays
using LinearAlgebra
using Symbolics
using Debugger
include("utils.jl")
include("newton.jl")
include("newton2.jl")

GRAVITY = 9.8
# pendulum model
M_ = @SMatrix [1.0 0 0; 0 1 0; 0 0 1]
J_ = Diagonal(@SVector[1.25, 0.33, 1])
com_to_anchor = [0, 0, 1]
com_to_alt_anchor = [0, 1, 0]
world_anchor = [0, 0, 0]
world_alt_anchor = [0, 1, 0]


function get_pendulum_state(θ)
    return [-sin(θ), 0, -cos(θ), cos(θ/2), 0, sin(θ/2), 0]
end

function constraint(q)
    pos, quat = q[1:3], q[4:7]
    rot = quat_to_rot(quat)
    anchor = pos + rot * com_to_anchor
    pos_constraint = world_anchor - anchor
    pos_constraint
end

function constraint_jac(q)
    pos, quat = q[1:3], q[4:7]
    rot = quat_to_rot(quat)
    cat(-Matrix(I, 3, 3), hat(rot * com_to_anchor), dims=2)
end

function constraint_jac_auto(q)
    ForwardDiff.jacobian(constraint, q) * world_attitude_jacobian_from_configs(q)
end

# function get_condition(q1, q2, Δt)
#     function condition(q3_and_λ)
#         q3, λ = q3_and_λ[1:length(q2)], q3_and_λ[length(q2)+1:end]
#         pos1, quat1 = q1[1:3], Quaternion(q1[4:7]...)
#         pos2, quat2 = q2[1:3], Quaternion(q2[4:7]...)
#         pos3, quat3 = q3[1:3], Quaternion(q3[4:7]...)
#         v1 = (pos2 - pos1) / Δt
#         v2 = (pos3 - pos2) / Δt
#         ωq1 = 2 * conj(quat1) * quat2 / Δt  # body frame
#         ωq2 = 2 * conj(quat2) * quat3 / Δt  # body frame
#         # ωq1 = 2 * quat2 * conj(quat1) / Δt  # inertial frame
#         # ωq2 = 2 * quat3 * conj(quat2) / Δt  # inertial frame
#         ω1 = [ωq1.v1, ωq1.v2, ωq1.v3] / ωq1.s
#         ω2 = [ωq2.v1, ωq2.v2, ωq2.v3] / ωq2.s
#         lin_cond = M_ * ((v2 - v1) / Δt + GRAVITY * [0, 0, 1])
#         ang_cond = J_ * ω2 * 2/Δt*√(1 - ω2'*ω2) + hat(ω2) * J_ * ω2 - J_ * ω1 * 2/Δt*√(1 - ω1'*ω1) + hat(ω1) * J_ * ω1
#         total_cond = cat(lin_cond, ang_cond, dims=1)
#         total_cond -= constraint_jac(q2)' * λ
#         cat(total_cond, constraint(q3), dims=1)
#     end
#     condition
# end

function get_condition(q1, q2, Δt)
    H = @SMatrix [0.0 0 0; 1 0 0; 0 1 0; 0 0 1]
    S = @SVector[1, 0, 0, 0]
    T = @SMatrix [1.0 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 -1]
    function condition(q3_and_λ)
        q3, λ = q3_and_λ[1:length(q2)], q3_and_λ[length(q2)+1:end]
        pos1, quat1 = q1[1:3], q1[4:7]
        pos2, quat2 = q2[1:3], q2[4:7]
        pos3, quat3 = q3[1:3], q3[4:7]
        v1 = (pos2 - pos1) / Δt
        v2 = (pos3 - pos2) / Δt
        lin_cond = M_ * ((v2 - v1) / Δt + GRAVITY * [0, 0, 1])
        ang_cond = -4/Δt^2 * ϕ(quat_L(quat1)' * quat2)' * J_ * (H' / (S' * quat_L(quat1)' * quat2) - H' * quat_L(quat1)' * quat2 * S' / (S' * quat_L(quat1)' * quat2)^2) * quat_L(quat1)' -
                    4/Δt^2 * ϕ(quat_L(quat2)' * quat3)' * J_ * (H' / (S' * quat_L(quat2)' * quat3) - H' * quat_L(quat2)' * quat3 * S' / (S' * quat_L(quat2)' * quat3)^2) * quat_R(quat3) * T
        total_cond = cat(lin_cond, body_attitude_jacobian(quat2)' * ang_cond', dims=1)
        total_cond -= constraint_jac(q2)' * λ
        cat(total_cond, constraint(q3), dims=1)
    end
    condition
end

function sim(q1, q2, Δt, time)
    """ ignores fixed config of first body """
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    num_constraints = 3  # 3 positional constraints for spherical joint
    λ = zeros(num_constraints)
    while t < time
        x0 = cat(q2, λ, dims=1)
        x = newton(get_condition(q1, q2, Δt), x0, quat_adjust=true, len_config=length(q2))
        q3 = x[1:length(q2)]
        λ = x[length(q2)+1:end]  # TODO try with and without this line
        push!(qs, q3)
        q1, q2 = q2, q3
        println("t=$(t), constr_norm=$(norm(constraint(q3)))")
        t += Δt
    end
    qs
end

function sim_sym(q1, q2, Δt, time)
    """ ignores fixed config of first body """
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    num_constraints = 3  # 3 positional constraints for spherical joint
    λ = zeros(num_constraints)
    while t < time
        x0 = cat(q2, λ, dims=1)
        cond_fn = get_condition(q1, q2, Δt)

        # symbolics
        @variables q3_and_λ[1:length(q2)+num_constraints]
        cond = cond_fn(q3_and_λ)
        cond_exp = Symbolics.build_function(cond, q3_and_λ)
        cond_func = eval(cond_exp[1])
        cond_jac = Symbolics.jacobian(cond, q3_and_λ, simplify=true)
        cond_jac_exp = Symbolics.build_function(cond_jac, q3_and_λ)
        cond_jac_fn = eval(cond_jac_exp[1])
        @bp

        # x = newton2_with_jac(cond_fn, x -> Base.invokelatest(cond_jac_fn, x), x0, len_config=length(q2), tol=1e-6, merit_norm=2)
        x = Base.invokelatest(newton2_with_jac, cond_fn, cond_jac_fn, x0, len_config=length(q2), tol=1e-6, merit_norm=2)
        q3 = x[1:length(q2)]
        λ = x[length(q2)+1:end]  # TODO try with and without this line
        push!(qs, q3)
        q1, q2 = q2, q3
        println("t=$(t), constr_norm=$(norm(constraint(q3)))")
        t += Δt
    end
    qs
end


q1 = get_pendulum_state(0)
q2 = get_pendulum_state(0.1)
num_constraints = 3
Δt = 0.01
cond_fn = get_condition(q1, q2, Δt)
λ = zeros(num_constraints)
x0 = cat(q2, λ, dims=1)

# symbolics
@variables q3_and_λ[1:length(q2)+num_constraints]
cond = cond_fn(q3_and_λ)
cond_exp = Symbolics.build_function(cond, q3_and_λ)
cond_func = eval(cond_exp[1])
cond_jac = Symbolics.jacobian(cond, q3_and_λ, simplify=true)
cond_jac_exp = Symbolics.build_function(cond_jac, q3_and_λ)
cond_jac_fn = eval(cond_jac_exp[1])
cond_jac_fn(x0)
println(cond_jac_fn(x0))

constraint_jac(q2) ≈ constraint_jac_auto(q2)

qs = sim(q1, q2, 0.01, 0.5)
qs_sym = sim_sym(q1, q2, 0.01, 0.5)

for q in qs[1:300]
    println(q)
end
