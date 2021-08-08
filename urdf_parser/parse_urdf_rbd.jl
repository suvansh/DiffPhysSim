import RigidBodyDynamics
RBD = RigidBodyDynamics
using LinearAlgebra
using BlockDiagonals
using DataStructures
using LightXML
using StaticArrays
using ForwardDiff
using Quaternions
using Symbolics
# using Roots
# using MeshCat, GeometryBasics, CoordinateTransformations
using Debugger
include("additional_joints.jl")
include("utils.jl")
include("newton.jl")

GRAVITATIONAL_ACCELERATION = 9.8

path_to_repo = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/"

singleton_path = path_to_repo*"urdf_parser/urdfs/singleton.urdf"
pendulum_path = path_to_repo*"urdf_parser/urdfs/pendulum.urdf"
acrobot_path = path_to_repo*"urdf_parser/urdfs/acrobot.urdf"
double_pendulum_path = path_to_repo*"urdf_parser/urdfs/double_pendulum.urdf"
cassie_path = path_to_repo*"urdf_parser/urdfs/cassie-old.urdf"
strandbeest_path = path_to_repo*"urdf_parser/urdfs/strandbeest.urdf"
bsm_path = path_to_repo*"urdf_parser/urdfs/bsm.urdf"
scissor_path = path_to_repo*"urdf_parser/urdfs/scissor_2d.urdf"


function add_loop_joints!(mechanism::RBD.Mechanism{T}, urdf::AbstractString) where T
    """
    taken from https://github.com/rdeits/StrandbeestRobot.jl/blob/master/src/StrandbeestRobot.jl
    """
    doc = parse_file(urdf)
    xml_root = LightXML.root(doc)
    xml_loops = get_elements_by_tagname(xml_root, "loop_joint")
    for xml_loop in xml_loops
        name = attribute(xml_loop, "name")
        @assert attribute(xml_loop, "type") == "continuous"
        axis = SVector{3}(RBD.parse_vector(T, find_element(xml_loop, "axis"), "xyz", "1 0 0"))
        joint = RBD.Joint(name, RBD.Revolute(axis))
        xml_link1 = find_element(xml_loop, "link1")
        # println(attribute(xml_link1, "link"))
        body1 = RBD.findbody(mechanism, attribute(xml_link1, "link"))
        H1 = RBD.Transform3D(RBD.frame_before(joint), RBD.default_frame(body1),
            RBD.parse_pose(T, xml_link1)...)
        xml_link2 = find_element(xml_loop, "link2")
        # println(attribute(xml_link2, "link"))
        body2 = RBD.findbody(mechanism, attribute(xml_link2, "link"))
        H2 = RBD.Transform3D(RBD.frame_after(joint), RBD.default_frame(body2),
            RBD.parse_pose(T, xml_link2)...)
        RBD.attach!(mechanism, body1, body2, joint,
            joint_pose = H1,
            successor_pose = inv(H2))
    end
end

get_bodies(mechanism::RBD.Mechanism) = RBD.bodies(mechanism)[2:end]  # exclude world body
get_joints(mechanism::RBD.Mechanism) = RBD.joints(mechanism)[2:end]  # exclude world joint

function constraint_vector(mechanism, state, body_order_dict)
    C = []
    for joint in get_joints(mechanism)
        parent_body = RBD.predecessor(joint, mechanism)
        child_body = RBD.successor(joint, mechanism)
        parent_idx = 7 * body_order_dict[parent_body.id] - 6
        child_idx = 7 * body_order_dict[child_body.id] - 6
        parent_com = check_mass(parent_body) ?
                        parent_body.inertia.cross_part / parent_body.inertia.mass :
                        zeros(3)
        child_com = child_body.inertia.cross_part / child_body.inertia.mass
        parent_state = state[parent_idx:parent_idx+6]
        child_state = state[child_idx:child_idx+6]

        """ positional constraint
        we take the difference between the anchor positions in world frame
        """
        # TODO get rid of rpy's for prismatic orientation constraint
        parent_rot = quat_to_rot(parent_state[4:7])
        parent_rpy = quat_to_rpy(parent_state[4:7])
        parent_pos = parent_state[1:3]
        child_rot = quat_to_rot(child_state[4:7])
        child_rpy = quat_to_rpy(child_state[4:7])
        child_pos = child_state[1:3]

        # com->joint = (orig->joint) - (orig->com)
        parent_com_to_joint = RBD.translation(RBD.joint_to_predecessor(joint)) - parent_com
        # TODO check if joint_to_successor is orig->joint (assumed below)
        # otherwise first term in child_com_to_joint needs to be negated (relevant for loop_joints)
        child_com_to_joint = RBD.translation(RBD.joint_to_successor(joint)) - child_com
        parent_anc_wf = parent_pos + parent_rot * parent_com_to_joint
        child_anc_wf = child_pos + child_rot * child_com_to_joint
        # println(parent_anc_wf, child_anc_wf)
        # dof pos and ori constraints
        pos_constraint = parent_anc_wf - child_anc_wf
        # println("parent: $(parent_pos)\n\t$(parent_rot)\n\t$(parent_com_to_joint)\n\t$(parent_com)")
        # println("child: $(child_pos),\n\t$(child_rot)\n\t$(child_com_to_joint)\n\t$(child_com)")
        # println()
        # TODO this prob needs to be x y z angles (not Euler but absolute)
        ori_constraint = parent_rpy - child_rpy

        """ joint-specific pos/ori constraints if not 3-dof constraints """
        if joint.joint_type isa RBD.Revolute
            """ orientation constraint
            we want a minimal (2d) representation of the 3d axis error
            take cross product of the world frame axes and project onto
            orthogonal complement of parent axis, finding coords in this plane
            """
            # TODO check if joint_to_successor is needed for rotation here, and whether it should be transposed
            # I assume so for loop_joint case (otherwise should be no rotation between origin and com?)
            child_frame_axis = RBD.rotation(RBD.joint_to_successor(joint))' * joint.joint_type.axis
            parent_frame_axis = RBD.rotation(RBD.joint_to_predecessor(joint))' * joint.joint_type.axis
            parent_axis_world_frame = parent_rot * parent_frame_axis
            child_axis_world_frame = child_rot * child_frame_axis
            parent_axis_world_perp = orthogonal_complement(parent_axis_world_frame)
            ori_constraint = parent_axis_world_perp * child_axis_world_frame

            # SS: quaternion derivative rederive note
            # parent_alt_anc_wf = parent_pos + parent_rot * (parent_com_to_joint + joint.joint_type.axis)
            # child_alt_anc_wf = child_pos + child_rot * (child_com_to_joint + joint.joint_type.axis)
            # ori_constraint_full = parent_alt_anc_wf - child_alt_anc_wf
            # ori_constraint = orthogonal_complement(parent_rot * joint.joint_type.axis) * ori_constraint_full
        elseif joint.joint_type isa RBD.Fixed
        elseif joint.joint_type isa Ball
            ori_constraint = zeros(0)  # no orientation constraint
        elseif joint.joint_type isa RBD.Prismatic
            parent_frame_axis = RBD.rotation(RBD.joint_to_predecessor(joint))' * joint.joint_type.axis
            parent_axis_world_frame = parent_rot * parent_frame_axis
            parent_axis_world_perp = orthogonal_complement(parent_axis_world_frame)
            pos_constraint = parent_axis_world_perp * pos_diff
        elseif joint.joint_type isa RBD.QuaternionFloating
            # no constraint
            continue
        else
            throw(DomainError(string(joint.joint_type), "unsupported joint type"))
        end
        push!(C, cat(pos_constraint, ori_constraint, dims=1))
    end
    isempty(C) ? C : cat(C..., dims=1)  # handle no joints case
end

function make_jac(jac_local, parent_body, child_body, body_order_dict)
    """ jac_local contains the jacobians for the two bodies.
    these must be laid out in the correct place for the relevant body idxs. """
    parent_jac_idx = 6 * body_order_dict[parent_body.id] - 5
    child_jac_idx = 6 * body_order_dict[child_body.id] - 5
    jac = zeros(eltype(jac_local), size(jac_local, 1), 6*length(body_order_dict))
    jac[:, parent_jac_idx:parent_jac_idx+5] = jac_local[:, 1:6]
    jac[:, child_jac_idx:child_jac_idx+5] = jac_local[:, 7:12]
    jac
end

function constraint_vector_jac(mechanism, state, body_order_dict)
    C = []
    for joint in get_joints(mechanism)
        parent_body = RBD.predecessor(joint, mechanism)
        child_body = RBD.successor(joint, mechanism)
        parent_idx = 7 * body_order_dict[parent_body.id] - 6
        child_idx = 7 * body_order_dict[child_body.id] - 6
        parent_com = check_mass(parent_body) ?
                        parent_body.inertia.cross_part / parent_body.inertia.mass :
                        zeros(3)
        child_com = child_body.inertia.cross_part / child_body.inertia.mass
        parent_state = state[parent_idx:parent_idx+6]
        child_state = state[child_idx:child_idx+6]

        """ positional constraint
        we take the difference between the anchor positions in world frame
        """
        parent_quat = parent_state[4:7]
        child_quat = child_state[4:7]
        parent_rot = quat_to_rot(parent_state[4:7])
        parent_rpy = quat_to_rpy(parent_state[4:7])
        parent_pos = parent_state[1:3]
        child_rot = quat_to_rot(child_state[4:7])
        child_rpy = quat_to_rpy(child_state[4:7])
        child_pos = child_state[1:3]

        # com->joint = (orig->joint) - (orig->com)
        parent_com_to_joint = RBD.translation(RBD.joint_to_predecessor(joint)) - parent_com  # body frame
        # TODO check if joint_to_successor is orig->joint (assumed below)
        # otherwise first term in child_com_to_joint needs to be negated (relevant for loop_joints)
        child_com_to_joint = RBD.translation(RBD.joint_to_successor(joint)) - child_com  # body frame
        parent_anc_wf = parent_pos + parent_rot * parent_com_to_joint
        child_anc_wf = child_pos + child_rot * child_com_to_joint
        # dof pos and ori constraints
        pos_constraint_jac = cat(Matrix(I, 3, 3), -hat(parent_rot * parent_com_to_joint), -Matrix(I, 3, 3), hat(child_rot * child_com_to_joint), dims=2)
        ori_constraint_jac = cat(zeros(3, 3), Matrix(I, 3, 3), zeros(3, 3), -Matrix(I, 3, 3), dims=2)

        """ joint-specific pos/ori constraints if not 3-dof constraints """
        if joint.joint_type isa RBD.Revolute
            """ orientation constraint
            we want a minimal (2d) representation of the 3d axis error
            take cross product of the world frame axes and project onto
            orthogonal complement of parent axis, finding coords in this plane
            """
            # TODO check if joint_to_successor is needed for rotation here, and whether it should be transposed
            # I assume so for loop_joint case (otherwise should be no rotation between origin and com?)
            child_frame_axis = RBD.rotation(RBD.joint_to_successor(joint))' * joint.joint_type.axis
            parent_frame_axis = RBD.rotation(RBD.joint_to_predecessor(joint))' * joint.joint_type.axis
            parent_axis_world_frame = parent_rot * parent_frame_axis
            child_axis_world_frame = child_rot * child_frame_axis
            parent_axis_world_perp = orthogonal_complement(parent_axis_world_frame)
            ori_constraint_jac = cat(zeros(2, 3), parent_axis_world_perp * hat(child_axis_world_frame), zeros(2, 3), -parent_axis_world_perp * hat(child_axis_world_frame), dims=2)

            # SS: quaternion derivative rederive note
            # parent_com_to_alt_anchor = parent_com_to_joint + joint.joint_type.axis
            # child_com_to_alt_anchor = child_com_to_joint + joint.joint_type.axis
            # parent_com_to_alt_anchor_wf = parent_rot * parent_com_to_alt_anchor
            # child_com_to_alt_anchor_wf = child_rot * child_com_to_alt_anchor
            # parent_alt_anc_wf = parent_pos + parent_com_to_alt_anchor_wf
            # child_alt_anc_wf = child_pos + child_com_to_alt_anchor_wf
            # oc = orthogonal_complement(parent_rot * joint.joint_type.axis)
            # # ori_constraint_jac = cat(oc, oc * (hat(parent_alt_anc_wf - child_alt_anc_wf) - hat(parent_com_to_alt_anchor_wf)), -oc, oc * hat(child_com_to_alt_anchor_wf), dims=2)
            # ori_constraint_jac = cat(oc,
            #     oc * hat(parent_alt_anc_wf - child_alt_anc_wf) + oc * H'*(L(L(parent_quat)'*H*parent_com_to_alt_anchor) + R(parent_quat)*R(H*parent_com_to_alt_anchor)*T) * world_attitude_jacobian(parent_quat),
            #     -oc,
            #     -oc * H'*(L(L(child_quat)'*H*child_com_to_alt_anchor) + R(child_quat)*R(H*child_com_to_alt_anchor)*T) * world_attitude_jacobian(child_quat),
            #     dims=2)
        elseif joint.joint_type isa RBD.Fixed
        elseif joint.joint_type isa Ball
            ori_constraint_jac = zeros(0, 12)  # no orientation constraint
        elseif joint.joint_type isa RBD.Prismatic
            child_frame_axis = RBD.rotation(RBD.joint_to_successor(joint))' * joint.joint_type.axis
            parent_frame_axis = RBD.rotation(RBD.joint_to_predecessor(joint))' * joint.joint_type.axis
            parent_axis_world_frame = parent_rot * parent_frame_axis
            child_axis_world_frame = child_rot * child_frame_axis
            child_axis_world_perp = orthogonal_complement(child_axis_world_frame)
            pos_constraint_jac = cat(child_axis_world_perp, -child_axis_world_perp * hat(parent_rot * parent_com_to_joint), -child_axis_world_perp, child_axis_world_perp * hat(pos_diff + child_rot * child_com_to_joint))
        elseif joint.joint_type isa RBD.QuaternionFloating
            # no constraint
            continue
        else
            throw(DomainError(string(joint.joint_type), "unsupported joint type"))
        end
        push!(C, make_jac(cat(pos_constraint_jac, ori_constraint_jac, dims=1), parent_body, child_body, body_order_dict))
    end
    isempty(C) ? C : cat(C..., dims=1)  # handle no joints case
end

function check_mass(body::RBD.RigidBody)
    RBD.has_defined_inertia(body) && body.inertia.mass != 0 && norm(body.inertia.moment) != 0
end

function get_mass_bodies(mechanism::RBD.Mechanism)
    filter(check_mass, get_bodies(mechanism))
end

function process_urdf(filename::String; floating=false, gravity=GRAVITATIONAL_ACCELERATION)
    joint_types = merge(Dict("ball" => Ball), RBD.default_urdf_joint_types())
    mechanism = RBD.parse_urdf(filename, joint_types=joint_types, remove_fixed_tree_joints=false,
                                floating=floating, gravity=SVector(0.0, 0.0, gravity))
    mass_bodies = get_mass_bodies(mechanism)
    body_enum = enumerate(get_bodies(mechanism))
    body_order_dict = Dict(b.id => i for (i, b) in body_enum)
    mass_body_idxs = [i for (i, b) in body_enum if check_mass(b)]
    # mass matrix
    mass_matrix = convert(Array{Float64}, BlockDiagonal(
        [BlockDiagonal([Matrix(body.inertia.mass*I, 3, 3), body.inertia.moment])
            for body in mass_bodies]))

    # add loop joints
    add_loop_joints!(mechanism, filename)

    # constraint function and Jacobian
    function constraint(state)
        constraint_vector(mechanism, state, body_order_dict)
    end

    function constraint_jac(state)
        constraint_vector_jac(mechanism, state, body_order_dict)
    end

    function constraint_jac_auto(state)
        jac = ForwardDiff.jacobian(constraint, state)
        jac * world_attitude_jacobian_from_configs(state)
    end

    function lagrangian(q1, q2, Δt)
        """
        discrete Lagrangian.
        implements Ld(q1, q2, λ, Δt) = Δt * L((q1+q2)/2, (q2-q1)/Δt),
        where q2-q1 involves some quaternion math
        """
        M = mass_matrix
        diff = []
        avg = []
        # chunk states into groups of 7 for each body
        q1_groups = collect(Iterators.partition(q1, 7))
        q2_groups = collect(Iterators.partition(q2, 7))
        # get the velocity and average config per body
        for (idx, (q1i, q2i)) in enumerate(zip(q1_groups, q2_groups))
            pos1, quat1 = q1i[1:3], Quaternion(q1i[4:7]...)
            pos2, quat2 = q2i[1:3], Quaternion(q2i[4:7]...)
            if idx ∈ mass_body_idxs
                v = pos2 - pos1
                ωq = 2 * conj(quat2) * quat1
                ω = [ωq.v1, ωq.v2, ωq.v3]
                push!(diff, cat(v, ω, dims=1))
            end
            pos_avg = (pos2 + pos1) / 2
            quat_avg = linpol(quat1, quat2, 0.5)  # TODO check this
            push!(avg, cat(pos_avg, [quat_avg.s, quat_avg.v1, quat_avg.v2, quat_avg.v3], dims=1))
            # push!(avg, (q1i+q2i)/2)
        end
        vel = cat(diff..., dims=1) / Δt
        qavg = cat(avg..., dims=1)
        # calculate Lagrangian
        T = 0.5*vel'*M*vel
        U = sum(body.inertia.mass * gravity * (q1i[3] + q2i[3])/2
                for (body, q1i, q2i) in zip(mass_bodies, q1_groups[mass_body_idxs], q2_groups[mass_body_idxs]))
        # C = constraint(q2)
        # Cλ = isempty(C) ? 0 : C'*λ  # λ ignored if no constraints
        Δt * (T - U)
    end

    function get_condition(q1, q2, Δt; force=false, torque=false, first_fixed=true)
        """ force and torque are arrays of length 3*num_bodies representing
        the external force and torque at timestep 2 """
        q1_groups = collect(Iterators.partition(q1, 7))
        q2_groups = collect(Iterators.partition(q2, 7))
        if force == false
            force = zeros(3*length(q1_groups))
        end
        if torque == false
            torque = zeros(3*length(q1_groups))
        end
        force_groups = collect(Iterators.partition(force, 3))
        torque_groups = collect(Iterators.partition(torque, 3))
        function condition(q3_and_λ)
            if first_fixed
                # first body is fixed, don't optimize over its config
                q3 = cat([0,0,0,1,0,0,0], q3_and_λ[1:length(q2)-7], dims=1)
                λ = q3_and_λ[length(q2)-7+1:end]
            else
                q3 = q3_and_λ[1:length(q2)]
                λ = q3_and_λ[length(q2)+1:end]
            end
            q3_groups = collect(Iterators.partition(q3, 7))
            total_cond = []
            for (body, q1i, q2i, q3i, force_i, torque_i) in zip(mass_bodies,
                    q1_groups[mass_body_idxs], q2_groups[mass_body_idxs],
                    q3_groups[mass_body_idxs], force_groups[mass_body_idxs],
                    torque_groups[mass_body_idxs])
                Mi, Ji = body.inertia.mass, body.inertia.moment
                pos1, quat1 = q1i[1:3], q1i[4:7]
                pos2, quat2 = q2i[1:3], q2i[4:7]
                pos3, quat3 = q3i[1:3], q3i[4:7]
                v1 = (pos2 - pos1) / Δt
                v2 = (pos3 - pos2) / Δt
                lin_cond = Mi * (v2 - v1 + GRAVITATIONAL_ACCELERATION * Δt * [0, 0, 1]) - force_i * Δt
                quat_1_to_2 = L(quat1)' * quat2
                quat_2_to_3 = L(quat2)' * quat3
                # ang_cond = -4/Δt * ϕ(quat_1_to_2)' * Ji * (H' / (S' * quat_1_to_2) - H' * quat_1_to_2 * S' / (S' * quat_1_to_2)^2) * L(quat1)' -
                #             4/Δt * ϕ(quat_2_to_3)' * Ji * (H' / (S' * quat_2_to_3) - H' * quat_2_to_3 * S' / (S' * quat_2_to_3)^2) * R(quat3) * T
                ang_cond = -4/Δt * L(quat1)' * H * Ji * H' * L(quat1)' * quat2 - 4/Δt * T * R(quat3)' * H * Ji * H' * R(quat3) * T * quat2
                # TODO this one below should be correct but the one above works way better
                # ang_cond = -4/Δt * L(quat1) * H * Ji * H' * L(quat1)' * quat2 - 4/Δt * T * R(quat3)' * H * Ji * H' * R(quat3) * T * quat2
                ang_cond = body_attitude_jacobian(quat2)' * ang_cond - 2 * torque_i * Δt
                total_cond_i = cat(lin_cond, ang_cond, dims=1)
                push!(total_cond, total_cond_i)
            end
            total_cond = cat(total_cond..., dims=1)
            total_cond -= Δt * constraint_jac(q2)[:,7:end]' * λ
            cat(total_cond, constraint(q3), dims=1)
        end
        condition
    end

    function get_condition_jacobian(q1, q2, Δt; first_fixed=true)
        q1_groups = collect(Iterators.partition(q1, 7))
        q2_groups = collect(Iterators.partition(q2, 7))
        function condition_jacobian(q3_and_λ)
            if first_fixed
                # first body is fixed, don't optimize over its config
                q3 = cat([0,0,0,1,0,0,0], q3_and_λ[1:length(q2)-7], dims=1)
                λ = q3_and_λ[length(q2)-7+1:end]
            else
                q3 = q3_and_λ[1:length(q2)]
                λ = q3_and_λ[length(q2)+1:end]
            end
            num_constraints = length(λ)
            q3_groups = collect(Iterators.partition(q3, 7))
            ∂cond_∂config_arr = []
            for (body, q1i, q2i, q3i) in zip(mass_bodies,
                    q1_groups[mass_body_idxs], q2_groups[mass_body_idxs],
                    q3_groups[mass_body_idxs])
                Mi = body.inertia.mass * Matrix(I, 3, 3)
                Ji = body.inertia.moment
                pos1, quat1 = q1i[1:3], q1i[4:7]
                pos2, quat2 = q2i[1:3], q2i[4:7]
                pos3, quat3 = q3i[1:3], q3i[4:7]
                ∂cond_∂pos = -cat(-Mi/Δt, zeros(3, 4), dims=2)
                term = 4/Δt * (T*L(H*Ji*H'*R(quat3)*T*quat2)*T + T*R(quat3)'*H*Ji*H'*L(quat2)')
                ∂cond_∂quat = -cat(zeros(3, 3), body_attitude_jacobian(quat2)' * term, dims=2)
                ∂cond_∂config_i = cat(∂cond_∂pos, ∂cond_∂quat, dims=1)
                push!(∂cond_∂config_arr, ∂cond_∂config_i)
            end
            ∂cond_∂config = convert(Array{Float64}, BlockDiagonal(convert(Array{Array{Float64, 2}}, ∂cond_∂config_arr)))
            jac_top = cat(∂cond_∂config * world_attitude_jacobian_from_configs(q3[8:end]), -Δt * constraint_jac(q2)'[7:end,:], dims=2)
            jac_constr = cat(constraint_jac(q3)[:,7:end], zeros(num_constraints, num_constraints), dims=2)
            jac = cat(jac_top, jac_constr, dims=1)
            # println(jac)
            jac
        end
        condition_jacobian
    end

    mechanism, mass_matrix, constraint, constraint_jac, constraint_jac_auto, lagrangian, get_condition, get_condition_jacobian
end

function simulate_unconstrained_system(q1, q2, Δt, lagrangian, time)
    D1(first, second) = convert(Array{Float64}, BlockDiagonal([Matrix(I, 3, 3), world_attitude_jacobian(first[4:7])]))' * ForwardDiff.gradient(first -> lagrangian(first, second, [], Δt), first)
    D2(first, second) = convert(Array{Float64}, BlockDiagonal([Matrix(I, 3, 3), world_attitude_jacobian(second[4:7])]))' * ForwardDiff.gradient(second -> lagrangian(first, second, [], Δt), second)
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    while t < time
        objective(q3) = D2(q1, q2) + D1(q2, q3)
        q3 = newton(objective, q2, quat_adjust=true, len_config=length(q2))
        println(objective(q3))
        push!(qs, q3)
        q1, q2 = q2, q3
        t += Δt
    end
    qs
end


function simulate_constrained_system_sym(q1, q2, Δt, get_condition, constraint_fn, num_constraints, time)
    """ uses Symbolics in place of ForwardDiff """
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    λ = zeros(num_constraints)
    while t < time
        x0 = cat(q2[8:end], λ, dims=1)  # ignore fixed config of first body
        cond_fn = get_condition(q1, q2, Δt, first_fixed=true)

        # symbolics
        @variables q3_and_λ[1:length(q2)-7+num_constraints]
        cond = cond_fn(q3_and_λ)
        cond_exp = Symbolics.build_function(cond, q3_and_λ, expression=Val{false})
        cond_func = eval(cond_exp[1])
        cond_jac = Symbolics.jacobian(cond, q3_and_λ, simplify=true)
        cond_jac_exp = Symbolics.build_function(cond_jac, q3_and_λ, expression=Val{false})
        cond_jac_fn = eval(cond_jac_exp[1])

        x = newton(cond_fn, x -> substitute(cond_jac_fn(x), Dict(q3_and_λ => x)),
                                x0, len_config=length(q2)-7, tol=1e-6, merit_norm=2)
        q3 = cat([0, 0, 0, 1, 0, 0, 0], x[1:length(q2)-7], dims=1)
        λ = x[length(q2)-7+1:end]  # TODO try with and without this line
        push!(qs, q3)
        q1, q2 = q2, q3
        println("t=$(t), constr_norm=$(norm(constraint_fn(q3)))")
        t += Δt
    end
    qs
end

function simulate_constrained_system(q1, q2, Δt, time, get_condition, get_condition_jacobian, constraint_fn, num_constraints; initial_force=false, initial_torque=false)
    """ uses analytic Jacobian in place of ForwardDiff """
    if initial_force == false
        initial_force = zeros(3 * length(q1) ÷ 7)
    end
    if initial_torque == false
        initial_torque = zeros(3 * length(q1) ÷ 7)
    end
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    num_timesteps = 2
    λ = zeros(num_constraints)
    while t < time
        x0 = cat(q2[8:end], λ, dims=1)  # ignore fixed config of first body
        if num_timesteps == 2
            cond_fn = get_condition(q1, q2, Δt, torque=initial_torque, force=initial_force)
        else
            cond_fn = get_condition(q1, q2, Δt)
        end
        cond_jac_fn = get_condition_jacobian(q1, q2, Δt)

        x = newton(cond_fn, cond_jac_fn, x0, tol=1e-14, len_config=length(q2)-7, ls_mult=0.4, num_iters=200)
        q3 = cat([0, 0, 0, 1, 0, 0, 0], x[1:length(q2)-7], dims=1)
        λ = x[length(q2)-7+1:end]  # TODO try with and without this line
        push!(qs, q3)
        q1, q2 = q2, q3
        println("t=$(t), constr_norm=$(norm(constraint_fn(q3)))")
        t += Δt
        num_timesteps += 1
    end
    qs
end


function get_test_state(mechanism::RBD.Mechanism; pos_offset=0)
    n = length(get_bodies(mechanism))
    z = []
    for i = 1:n
        pos = rand(3) .+ pos_offset
        quat = rand(4)
        quat /= norm(quat)
        push!(z, cat(pos, quat, dims=1))
    end
    cat(z..., dims=1)
end

function transform_to_rot(transform)
    """ takes in 4x4 transform, returns 3x3 rotation component """
    [[transform[1] transform[2] transform[3]];
    [transform[5] transform[6] transform[7]];
    [transform[9] transform[10] transform[11]]]
end

function get_body_coordinates(body::RBD.RigidBody, mechanism::RBD.Mechanism)
    """ returns coordinates of body's CoM """
    parent_joint = RBD.joint_to_parent(body, mechanism)
    parent_body = RBD.predecessor(parent_joint, mechanism)
    total_transform = RBD.joint_to_successor(parent_joint).mat * RBD.joint_to_predecessor(parent_joint).mat
    total_translation = RBD.translation(RBD.joint_to_successor(parent_joint)) + RBD.translation(RBD.joint_to_predecessor(parent_joint))
    # println("translation so far ", total_translation)
    while parent_body !== RBD.root_body(mechanism)
        # println(parent_body)
        parent_joint = RBD.joint_to_parent(parent_body, mechanism)
        transform = RBD.joint_to_predecessor(parent_joint)
        total_transform *= transform.mat
        # total_rotation = transform_to_rot(total_transform)
        total_translation = RBD.translation(transform) + transform_to_rot(transform.mat) \ total_translation
        # println("translation so far ", total_translation)
        # println("rotation so far ", total_rotation)
        parent_body = RBD.predecessor(parent_joint, mechanism)
    end
    # total_rotation is world to body rotation
    total_rotation = transform_to_rot(total_transform)
    body_to_CoM = total_rotation \ (body.inertia.cross_part / body.inertia.mass)
    convert(Array{Float64}, total_translation + body_to_CoM)
end

function get_default_state(mechanism::RBD.Mechanism; start_index=1)
    """ gets initial configuration of a Mechanism starting at the START_INDEXth body """
    cat([0, 0, 0, 1, 0, 0, 0],
        [cat(get_body_coordinates(body, mechanism), [1, 0, 0, 0], dims=1)
            for body in get_bodies(mechanism)[start_index:end]]...,
        dims=1
    )
end


# # testing get_body_coordinates
# b1 = get_bodies(mechanism)[12]
# j1 = RBD.joint_to_parent(b1, mechanism)
# RBD.joint_to_successor(j1)
# RBD.joint_to_predecessor(j1)
# get_body_coordinates(get_bodies(mechanism)[2], mechanism)
# length(get_bodies(mechanism))

function get_torque(mechanism, body_idx1, body_idx2, torque_per_body)
    """ applies torque to a joint by applying opposite-sign torques at the specified links """
    torque = zeros(3*length(get_bodies(mechanism)))
    torque[3*body_idx1-2:3*body_idx1] = torque_per_body
    torque[3*body_idx2-2:3*body_idx2] = -torque_per_body
    torque
end

function get_acrobot_state(θ₁, θ₂)
    return [0, 0, 0, 1, 0, 0, 0,  # fixed
            -0.5sin(θ₁), 0.15, -0.5cos(θ₁), cos(0.5θ₁), 0, sin(0.5θ₁), 0,  # upper link
            -sin(θ₁)-sin(θ₁+θ₂), 0.25, -cos(θ₁)-cos(θ₁+θ₂), cos(0.5(θ₁+θ₂)), 0, sin(0.5(θ₁+θ₂)), 0]  # lower link
end
get_pendulum_state(θ) = get_acrobot_state(θ, 0)[1:14]

function get_double_pendulum_state(θ₁, θ₂)
    return [0, 0, 0, 1, 0, 0, 0,  # fixed
            -sin(θ₁), 0, -cos(θ₁), cos(θ₁/2), 0, sin(θ₁/2), 0,  # upper link
            -2sin(θ₁)-sin(θ₁+θ₂), 0.1, -2cos(θ₁)-cos(θ₁+θ₂), cos((θ₁+θ₂)/2), 0, sin((θ₁+θ₂)/2), 0]  # lower link
end


function get_first_configs(path)
    if path == acrobot_path
        q0 = get_acrobot_state(0, 0)
        q1 = get_acrobot_state(0.02, 0.02)
    elseif path == double_pendulum_path
        q0 = get_double_pendulum_state(0, 0)
        q1 = get_double_pendulum_state(0.02, 0.02)
    elseif path == singleton_path
        q0 = cat(zeros(3), [1, 0, 0, 0], dims=1)
        q1 = cat(zeros(3), [0.999, 0.0, 0.0436, 0.0001], dims=1)
    elseif path == pendulum_path
        q0 = get_pendulum_state(0)
        q1 = get_pendulum_state(0.05)
    else
        q0 = get_default_state(mechanism, start_index=2)  # ignore first fixed body
        q1 = q0  # will use torque instead of perturbed state to induce motion
    end
    return q0, q1
end


function fd_jac(state, constraint_fn, constraint_jac; ϵ=0.001)
    ∂C_∂q = []  # finite diff jacobian
    c₀ = constraint_fn(state)
    for i = 1:length(state)
        state_left, state_right = copy(state), copy(state)
        state_left[i] -= ϵ/2
        state_right[i] += ϵ/2
        c_leftᵢ = constraint_fn(state_left)
        c_rightᵢ = constraint_fn(state_right)
        ∂C_∂qᵢ = (c_rightᵢ-c_leftᵢ)/ϵ
        push!(∂C_∂q, ∂C_∂qᵢ)
    end
    ∂C_∂q = cat(∂C_∂q..., dims=2)
    ∂C_∂q, (∂C_∂q * world_attitude_jacobian_from_configs(state))
end



### SIMULATION
# NOTE change active_path between scissor_path, bsm_path, and double_pendulum_path
active_path = bsm_path

mechanism, mass_matrix, constraint_fn, constraint_jac, constraint_jac_auto, lagrangian, get_condition, get_condition_jacobian = process_urdf(active_path, floating=false)
num_constraints = length(constraint_fn(get_test_state(mechanism)))
q0, q1 = get_first_configs(active_path)

if active_path == bsm_path
    qs = simulate_constrained_system(q0, q1, 0.01, 1, get_condition,
                get_condition_jacobian, constraint_fn, num_constraints,
                initial_torque=get_torque(mechanism, 4, 12, [0, 0.01, 0]))
elseif active_path == scissor_path
    qs = simulate_constrained_system(q0, q1, 0.01, 1, get_condition,
                get_condition_jacobian, constraint_fn, num_constraints,
                initial_torque=get_torque(mechanism, 3, 4, [0, 0.01, 0]))
else
    qs = simulate_constrained_system(q0, q1, 0.01, 1, get_condition,
                get_condition_jacobian, constraint_fn, num_constraints)
end


# vis = Visualizer()
# delete!(vis)
# render(vis)
#
# # axis_case ∈ ["maximal", "intermediate", "minimal"]
# axis_case = "maximal"
# if axis_case == "maximal"
#     # singleton.urdf should have Ixx = 1.2, Iyy = 2.4, Izz = 0.8
#     # <inertia ixx="1.2" ixy="0" ixz="0" iyy="2.4" iyz="0" izz="0.8" />
#     box_dims = [0.4, 0.2, 0.6]
# elseif axis_case == "intermediate"
#     # singleton.urdf should have Ixx = 2.4, Iyy = 1.2, Izz = 0.8
#     # <inertia ixx="2.4" ixy="0" ixz="0" iyy="1.2" iyz="0" izz="0.8" />
#     box_dims = [0.2, 0.4, 0.6]
# elseif axis_case == "minimal"
#     # singleton.urdf should have Ixx = 1.2, Iyy = 0.8, Izz = 2.4
#     # <inertia ixx="1.2" ixy="0" ixz="0" iyy="0.8" iyz="0" izz="2.4" />
#     box_dims = [0.4, 0.6, 0.2]
# else
#     throw(DomainError(string(axis_case), "unsupported axis case"))
# end
#
# # animate
# setobject!(vis[:link], Rect(Vec(-box_dims./2...), Vec(box_dims...)))
# anim = Animation()
# for i = 1:length(qs)
#     pos_i, quat_i = qs[i][1:3], Quaternion(qs[i][4:7]...)
#     angle_i, axis_i = angleaxis(quat_i)
#     atframe(anim, i) do
#         settransform!(vis[:link], compose(Translation(qs[i][1:3]), LinearMap(AngleAxis(angle_i, axis_i...))))
#     end
# end
# setanimation!(vis, anim)
