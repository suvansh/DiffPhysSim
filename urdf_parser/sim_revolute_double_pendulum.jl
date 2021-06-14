""" 2 2m links forming revolute double pendulum with anchor at origin """
using Quaternions
using StaticArrays
using LinearAlgebra
using Symbolics
include("utils.jl")
include("newton.jl")
include("newton2.jl")
using Debugger

GRAVITY = 9.8
# pendulum model
y_offset = 0.1  # y offset between links
M_ = @SMatrix [1.0 0 0; 0 1 0; 0 0 1]  # shared by both links
J_ = Diagonal(@SVector[2.0, 1.33, 1])  # shared by both links
com_1_to_anchor_1 = [0, 0, 1]
world_anchor_1 = [0, 0, 0]  # anchor of body 1 in world frame
com_1_to_anchor_2 = [0, y_offset, -1]
com_2_to_anchor_2 = [0, 0, 1]
world_axis_1 = [0, 1, 0]  # fixed axis 1 in world ("parent") frame
axis_1_body_1 = [0, 1, 0]  # initial axis 1 in body 1 frame
axis_2_body_1 = [0, 1, 0]  # initial axis 2 in body 1 frame
axis_2_body_2 = [0, 1, 0]  # initial axis 2 in body 2 frame

function cget_double_pendulum_state(θ₁, θ₂)
    return [-sin(θ₁), 0, -cos(θ₁), cos(θ₁/2), 0, sin(θ₁/2), 0,  # upper link
            -2sin(θ₁)-sin(θ₁+θ₂), y_offset, -2cos(θ₁)-cos(θ₁+θ₂), cos((θ₁+θ₂)/2), 0, sin((θ₁+θ₂)/2), 0]  # lower link
end

function cconstraint(q)
    pos1, quat1 = q[1:3], q[4:7]
    pos2, quat2 = q[8:10], q[11:14]
    rot1 = quat_to_rot(quat1)
    rot2 = quat_to_rot(quat2)
    anchor_1 = pos1 + rot1 * com_1_to_anchor_1
    anchor_2_body_1 = pos1 + rot1 * com_1_to_anchor_2
    anchor_2_body_2 = pos2 + rot2 * com_2_to_anchor_2
    pos_constraint_1 = world_anchor_1 - anchor_1
    pos_constraint_2 = anchor_2_body_1 - anchor_2_body_2

    world_axis_1_perp = orthogonal_complement(world_axis_1)
    axis_1_body_1_world_frame = rot1 * axis_1_body_1
    ori_constraint_1 = world_axis_1_perp * axis_1_body_1_world_frame

    axis_2_body_1_world_frame = rot1 * axis_2_body_1
    axis_2_body_1_perp = orthogonal_complement(axis_2_body_1_world_frame)
    axis_2_body_2_world_frame = rot2 * axis_2_body_2
    ori_constraint_2 = axis_2_body_1_perp * axis_2_body_2_world_frame
    cat(pos_constraint_1, ori_constraint_1, pos_constraint_2, ori_constraint_2, dims=1)
end

function cconstraint_jac_auto(q)
    ForwardDiff.jacobian(cconstraint, q) * world_attitude_jacobian_from_configs(q)
end

function cconstraint_jac(q)
    pos1, quat1 = q[1:3], q[4:7]
    pos2, quat2 = q[8:10], q[11:14]
    rot1 = quat_to_rot(quat1)
    rot2 = quat_to_rot(quat2)

    pos_jac_1 = cat(-Matrix(I, 3, 3), hat(rot1 * com_1_to_anchor_1), zeros(3, 6), dims=2)
    pos_jac_2 = cat(Matrix(I, 3, 3), -hat(rot1 * com_1_to_anchor_2), -Matrix(I, 3, 3), hat(rot2 * com_2_to_anchor_2), dims=2)

    world_axis_1_perp = orthogonal_complement(world_axis_1)
    axis_1_body_1_world_frame = rot1 * axis_1_body_1
    ori_jac_1 = cat(zeros(2, 3), -world_axis_1_perp * hat(axis_1_body_1_world_frame), zeros(2, 6), dims=2)

    axis_2_body_1_world_frame = rot1 * axis_2_body_1
    axis_2_body_1_perp = orthogonal_complement(axis_2_body_1_world_frame)
    axis_2_body_2_world_frame = rot2 * axis_2_body_2
    ori_jac_2 = cat(zeros(2, 3), axis_2_body_1_perp * hat(axis_2_body_2_world_frame), zeros(2, 3), -axis_2_body_1_perp * hat(axis_2_body_2_world_frame), dims=2)

    cat(pos_jac_1, ori_jac_1, pos_jac_2, ori_jac_2, dims=1)
end

function cget_condition(q1, q2, Δt)
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
            quat_1_to_2 = quat_L(quat1)' * quat2
            quat_2_to_3 = quat_L(quat2)' * quat3
            ang_cond = -4/Δt * ϕ(quat_1_to_2)' * J_ * (H' / (S' * quat_1_to_2) - H' * quat_1_to_2 * S' / (S' * quat_1_to_2)^2) * quat_L(quat1)' -
                        4/Δt * ϕ(quat_2_to_3)' * J_ * (H' / (S' * quat_2_to_3) - H' * quat_2_to_3 * S' / (S' * quat_2_to_3)^2) * quat_R(quat3) * T
            total_cond_i = cat(lin_cond, body_attitude_jacobian(quat2)' * ang_cond', dims=1)
            push!(total_cond, total_cond_i)
        end
        total_cond = cat(total_cond..., dims=1)
        total_cond -= Δt * cconstraint_jac(q2)' * λ
        cat(total_cond, cconstraint(q3), dims=1)
    end
    condition
end

function csim(q1, q2, Δt, time)
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    num_constraints = 10  # 5 constraints per revolute joint
    λ = zeros(num_constraints)
    while t < time
        x0 = cat(q2, λ, dims=1)
        x = newton2(cget_condition(q1, q2, Δt), x0, len_config=length(q2), ls_mult=0.4, merit_norm=2, num_iters=35)
        q3 = x[1:length(q2)]
        λ = x[length(q2)+1:end]  # TODO try with and without this line
        push!(qs, q3)
        q1, q2 = q2, q3
        println("t=$(t), constr_norm=$(norm(cconstraint(q3)))")
        t += Δt
    end
    qs
end

function csim_sym(q1, q2, Δt, time)
    """ ignores fixed config of first body """
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    num_constraints = 10  # 5 constraints per revolute joint
    λ = zeros(num_constraints)
    while t < time
        x0 = cat(q2, λ, dims=1)
        cond_fn = cget_condition(q1, q2, Δt)

        # symbolics
        @variables q3_and_λ[1:length(q2)+num_constraints]
        cond = cond_fn(q3_and_λ)
        cond_exp = Symbolics.build_function(cond, q3_and_λ)
        cond_func = eval(cond_exp[1])
        @bp
        cond_jac = Symbolics.jacobian(cond, q3_and_λ, simplify=true)
        cond_jac_exp = Symbolics.build_function(cond_jac, q3_and_λ)
        cond_jac_fn = eval(cond_jac_exp[1])

        # x = newton2_with_jac(cond_fn, cond_jac_fn, x0, len_config=length(q2), tol=1e-6, merit_norm=2)
        # need Base.invokelatest because of above call to `eval` causes an error to do with world age
        # see discussion @ https://discourse.julialang.org/t/how-to-bypass-the-world-age-problem/7012/23
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


q1 = cget_double_pendulum_state(0, 0)
q2 = cget_double_pendulum_state(0.02, 0.02)
q3 = cget_double_pendulum_state(0.5, 0.16)


cqs = csim(q1, q2, 0.03, 1)
cqs_sym = csim_sym(q1, q2, 0.03, 1)
cqs[3]

# comparing urdf to non urdf versions
test_idx = 8
cconstraint_jac(cqs[test_idx])[:,7:12]≈constraint_jac(cat([0,0,0,1,0,0,0],cqs[test_idx], dims=1))[:,13:18]
cconstraint_jac(cqs[test_idx])[:,7:12]-constraint_jac(qs[test_idx])[:,13:18]
cqs[test_idx] - qs[test_idx][8:end]
[norm(qi[8:end]-cqi) for (qi, cqi) in zip(qs, cqs)]
[norm(cconstraint(qi)) for qi in cqs]
[norm(constraint_fn(cat([0,0,0,1,0,0,0], qi, dims=1))) for qi in cqs]

# this shows the constraint functions and jacs are the same
[norm(constraint_fn(cat([0,0,0,1,0,0,0], qi, dims=1))-cconstraint(qi)) for qi in cqs]
[norm(constraint_jac(qi)[:,7:end]-cconstraint_jac(qi[8:end])) for qi in qs]

constraint_jac(qs[2])[:,1:6],constraint_jac_auto(qs[2])[:,1:6]

for i in 1:length(qs)
    if !(cconstraint_jac(qs[i])[:,1:6]≈cconstraint_jac_auto(qs[i])[:,1:6])
        println(i)
    end
end

for q in qs
    println(q)
end

qs_stack = cat(qs..., dims=2)'
print(qs_stack[:,1])
