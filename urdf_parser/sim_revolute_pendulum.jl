""" 2m spherical pendulum with anchor at origin """
using Quaternions
using StaticArrays
using LinearAlgebra
using Symbolics
include("utils.jl")
include("newton.jl")
include("newton2.jl")

GRAVITY = 9.8
# pendulum model
M_ = @SMatrix [1.0 0 0; 0 1 0; 0 0 1]
J_ = Diagonal(@SVector[1.25, 0.33, 1])
com_to_anchor = [0, 0, 1]
world_anchor = [0, 0, 0]
world_axis = [0, 1, 0]  # fixed axis in world ("parent") frame
child_axis = [0, 1, 0]  # initial axis in child frame
world_alt_anchor = [0, 1, 0]  # point along axis
com_to_alt_anchor = [0, 1, 1]  # com to same point along axis

function get_pendulum_state(θ)
    return [-sin(θ), 0, -cos(θ), cos(θ/2), 0, sin(θ/2), 0]
end

# function constraint(q)
#     pos, quat = q[1:3], q[4:7]
#     rot = quat_to_rot(quat)
#     anchor = pos + rot * com_to_anchor
#     pos_constraint = world_anchor - anchor
#     world_axis_perp = orthogonal_complement(world_axis)
#     child_axis_world_frame = rot * child_axis
#     ori_constraint = world_axis_perp * child_axis_world_frame
#     cat(pos_constraint, ori_constraint, dims=1)
# end

function constraint(q)
    pos, quat = q[1:3], q[4:7]
    rot = quat_to_rot(quat)
    anchor = pos + rot * com_to_anchor
    pos_constraint = world_anchor - anchor
    alt_anchor = pos + rot * com_to_alt_anchor
    rev_constraint_full = world_alt_anchor - alt_anchor
    rev_constraint = orthogonal_complement(world_alt_anchor - world_anchor) * rev_constraint_full
    cat(pos_constraint, rev_constraint, dims=1)
end

# function constraint_jac(q)
#     pos, quat = q[1:3], q[4:7]
#     rot = quat_to_rot(quat)
#     world_axis_perp = orthogonal_complement(world_axis)
#     child_axis_world_frame = rot * child_axis
#     pos_jac = cat(-Matrix(I, 3, 3), hat(rot * com_to_anchor), dims=2)
#     ori_jac = cat(zeros(2, 3), -world_axis_perp * hat(child_axis_world_frame), dims=2)
#     cat(pos_jac, ori_jac, dims=1)
# end

function constraint_jac(q)
    pos, quat = q[1:3], q[4:7]
    rot = quat_to_rot(quat)
    pos_jac = cat(-Matrix(I, 3, 3), hat(rot * com_to_anchor), dims=2)

    oc = orthogonal_complement(world_alt_anchor - world_anchor)
    rev_jac = cat(-oc, oc * hat(rot * com_to_alt_anchor), dims=2)

    cat(pos_jac, rev_jac, dims=1)
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
#         ω1 = [ωq1.v1, ωq1.v2, ωq1.v3]
#         ω2 = [ωq2.v1, ωq2.v2, ωq2.v3]
#         # eq 26 from Jan paper
#         lin_cond = M_ * ((v2 - v1) / Δt + GRAVITY * [0, 0, 1])
#         # eq 33
#         ang_cond = J_ * ω2 * √(4/Δt^2 - ω2'*ω2) + hat(ω2) * J_ * ω2 - J_ * ω1 * √(4/Δt^2 - ω1'*ω1) + hat(ω1) * J_ * ω1
#         total_cond = cat(lin_cond, ang_cond, dims=1)
#         total_cond -= constraint_jac(q2)' * λ
#         cat(total_cond, constraint(q3), dims=1)
#     end
#     condition
# end

function get_condition(q1, q2, Δt)
    q1_groups = collect(Iterators.partition(q1, 7))
    q2_groups = collect(Iterators.partition(q2, 7))
    H = @SMatrix [0.0 0 0; 1 0 0; 0 1 0; 0 0 1]
    S = @SVector[1, 0, 0, 0]
    T = @SMatrix [1.0 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 -1]
    function condition(q3_and_λ)
        q3, λ = q3_and_λ[1:length(q2)], q3_and_λ[length(q2)+1:end]
        q3_groups = collect(Iterators.partition(q3, 7))
        total_cond = []
        for (q1i, q2i, q3i) in zip(q1_groups, q2_groups, q3_groups)
            pos1, quat1 = q1i[1:3], q1i[4:7]
            pos2, quat2 = q2i[1:3], q2i[4:7]
            pos3, quat3 = q3i[1:3], q3i[4:7]
            v1 = (pos2 - pos1) / Δt
            v2 = (pos3 - pos2) / Δt
            lin_cond = M_ * (v2 - v1 + GRAVITY * Δt * [0, 0, 1])
            ang_cond = -4/Δt * ϕ(quat_L(quat1)' * quat2)' * J_ * (H' / (S' * quat_L(quat1)' * quat2) - H' * quat_L(quat1)' * quat2 * S' / (S' * quat_L(quat1)' * quat2)^2) * quat_L(quat1)' -
                        4/Δt * ϕ(quat_L(quat2)' * quat3)' * J_ * (H' / (S' * quat_L(quat2)' * quat3) - H' * quat_L(quat2)' * quat3 * S' / (S' * quat_L(quat2)' * quat3)^2) * quat_R(quat3) * T
            total_cond_i = cat(lin_cond, body_attitude_jacobian(quat2)' * ang_cond', dims=1)
            push!(total_cond, total_cond_i)
        end
        total_cond = cat(total_cond..., dims=1)
        total_cond -= Δt * constraint_jac_auto(q2)' * λ
        cat(total_cond, constraint(q3), dims=1)
    end
    condition
end

function sim(q1, q2, Δt, time)
    """ ignores fixed config of first body """
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    num_constraints = 5  # 3 positional, 2 orientation constraints for revolute joint
    λ = zeros(num_constraints)
    while t < time
        x0 = cat(q2, λ, dims=1)
        x = newton2(get_condition(q1, q2, Δt), x0, len_config=length(q2), tol=1e-6, merit_norm=2)
        q3 = x[1:length(q2)]
        # λ = x[length(q2)+1:end]  # TODO try with and without this line
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
    num_constraints = 5  # 5 constraints per revolute joint
    λ = zeros(num_constraints)
    @bp
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

        x = newton2_with_jac(cond_fn, x -> Base.invokelatest(cond_jac_fn, x), x0, len_config=length(q2), tol=1e-6, merit_norm=2)
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
q2 = get_pendulum_state(0.15)

function constraint_jac_fd(state; ϵ=0.001)
    ∂C_∂q = []  # finite diff jacobian
    c₀ = constraint(state)
    for i = 1:length(state)
        state_left, state_right = copy(state), copy(state)
        state_left[i] -= ϵ/2
        state_right[i] += ϵ/2
        c_leftᵢ = constraint(state_left)
        c_rightᵢ = constraint(state_right)
        ∂C_∂qᵢ = (c_rightᵢ-c_leftᵢ)/ϵ
        push!(∂C_∂q, ∂C_∂qᵢ)
    end
    ∂C_∂q = cat(∂C_∂q..., dims=2)
    ∂C_∂q * world_attitude_jacobian_from_configs(state)
end

constraint_jac(q2) ≈ constraint_jac_auto(q2) && constraint_jac(q2) ≈ constraint_jac_fd(q2)

qs = sim(q1, q2, 0.1, 5)
qs_sym = sim_sym(q1, q2, 0.1, 5)

for q in qs
    println(q)
end
