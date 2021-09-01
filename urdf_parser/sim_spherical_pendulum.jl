""" 2m spherical pendulum with anchor at origin """
using Quaternions
using StaticArrays
using LinearAlgebra
# using Debugger
include("utils.jl")
include("newton.jl")

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
    function condition(q3_and_λ)
        q3, λ = q3_and_λ[1:length(q2)], q3_and_λ[length(q2)+1:end]
        pos1, quat1 = q1[1:3], q1[4:7]
        pos2, quat2 = q2[1:3], q2[4:7]
        pos3, quat3 = q3[1:3], q3[4:7]
        v1 = (pos2 - pos1) / Δt
        v2 = (pos3 - pos2) / Δt
        lin_cond = M_ * (v2 - v1 + Δt * GRAVITY * [0, 0, 1])
        # ang_cond = -4/Δt * ϕ(L(quat1)' * quat2)' * J_ * (H' / (S' * L(quat1)' * quat2) - H' * L(quat1)' * quat2 * S' / (S' * L(quat1)' * quat2)^2) * L(quat1)' -
        #             4/Δt * ϕ(L(quat2)' * quat3)' * J_ * (H' / (S' * L(quat2)' * quat3) - H' * L(quat2)' * quat3 * S' / (S' * L(quat2)' * quat3)^2) * R(quat3) * T
        ang_cond = (-4/Δt * L(quat1)' * H * J_ * H' * L(quat1)' * quat2 - 4/Δt * T * R(quat3)' * H * J_ * H' * R(quat3) * T * quat2)'
        total_cond = cat(lin_cond, body_attitude_jacobian(quat2)' * ang_cond', dims=1)
        total_cond -= Δt * constraint_jac(q2)' * λ
        cat(total_cond, constraint(q3), dims=1)
    end
    condition
end

function get_condition_jacobian(q1, q2, Δt)
    function condition_jacobian(q3_and_λ)
        q3, λ = q3_and_λ[1:length(q2)], q3_and_λ[length(q2)+1:end]
        num_constraints = length(λ)
        pos1, quat1 = q1[1:3], q1[4:7]
        pos2, quat2 = q2[1:3], q2[4:7]
        pos3, quat3 = q3[1:3], q3[4:7]
        ∂cond_∂pos = -cat(-M_/Δt, zeros(3, 4), dims=2)
        term = 4/Δt * (T*L(H*J_*H'*R(quat3)*T*quat2)*T + T*R(quat3)'*H*J_*H'*L(quat2)')
        ∂cond_∂quat = -cat(zeros(3, 3), body_attitude_jacobian(quat2)' * term, dims=2)
        ∂cond_∂config = cat(∂cond_∂pos, ∂cond_∂quat, dims=1)
        jac_top = cat(∂cond_∂config * world_attitude_jacobian_from_configs(q3), -Δt * constraint_jac(q2)', dims=2)
        jac_constr = cat(constraint_jac(q3), zeros(num_constraints, num_constraints), dims=2)
        cat(jac_top, jac_constr, dims=1)
    end
    condition_jacobian
end

function sim(q1, q2, Δt, time)
    """ ignores fixed config of first body """
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    num_constraints = 3  # 3 positional constraints for spherical joint
    λ = zeros(num_constraints)
    while t < time
        x0 = cat(q2, λ, dims=1)
        x = newton(get_condition(q1, q2, Δt), x0, tol=1e-14, quat_adjust=true, len_config=length(q2))
        q3 = x[1:length(q2)]
        λ = x[length(q2)+1:end]  # TODO try with and without this line
        push!(qs, q3)
        q1, q2 = q2, q3
        println("t=$(t), constr_norm=$(norm(constraint(q3)))")
        t += Δt
    end
    qs
end

function sim_man(q1, q2, Δt, time)
    """ uses manual jacobian of condition """
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    num_constraints = 3  # 3 positional constraints for spherical joint
    λ = zeros(num_constraints)
    while t < time
        x0 = cat(q2, λ, dims=1)
        cond_fn = get_condition(q1, q2, Δt)
        cond_jac_fn = get_condition_jacobian(q1, q2, Δt)

        x = newton2_with_jac(cond_fn, cond_jac_fn, x0, apply_attitude=false, len_config=length(q2), tol=1e-14, merit_norm=2)
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

# test resolve_constraint
q_test = copy(q1)
# q_test[3] = -3  # -z translation by 2m (should be -1). works
q_test[3:end] = [-3, 0.1246747, 0, 0, 0.9921977]  # -z translation and significant rotation about the x-axis (by 0.25 rad). works
q_adj = resolve_constraint(q_test, constraint, constraint_jac, num_constraints, α=1e-4, tol=1e-10, ls=false)
# check the right-hand side vector (which has two pieces)
world_attitude_jacobian_from_configs(q_adj)' * config_diff(q_test, q_adj)
constraint(q_adj)

cond_fn = get_condition(q1, q2, Δt)
cond_jac_fn = get_condition_jacobian(q1, q2, Δt)
test_state = cat(q2, zeros(3), dims=1)
using ForwardDiff
fd_jac = ForwardDiff.jacobian(cond_fn, test_state)
fd_jac = cat(fd_jac[:, 1:7] * world_attitude_jacobian_from_configs(q2), fd_jac[:, 8:end], dims=2)
fd_man = cond_jac_fn(test_state)
fd_jac ≈ fd_man

λ = zeros(num_constraints)
x0 = cat(q2, λ, dims=1)


constraint_jac(q2) ≈ constraint_jac_auto(q2)

qs = sim(q1, q2, 0.01, 0.5)
qs_man = sim_man(q1, q2, 0.01, 0.5)

for q in qs[1:300]
    println(q)
end
