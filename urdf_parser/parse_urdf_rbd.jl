import RigidBodyDynamics
RBD = RigidBodyDynamics
using LinearAlgebra
using BlockDiagonals
using DataStructures
using LightXML
using StaticArrays
using ForwardDiff
using Quaternions
using Roots
using MeshCat, GeometryBasics, CoordinateTransformations
include("utils.jl")
include("newton.jl")

singleton_path = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/singleton.urdf"
acrobot_path = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/acrobot.urdf"
cassie_path = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/cassie-old.urdf"
strandbeest_path = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/strandbeest.urdf"

GRAVITATIONAL_ACCELERATION = 0#9.81

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
        # dof pos and ori constraints
        pos_diff = parent_anc_wf - child_anc_wf
        # println("parent: $(parent_pos)\n\t$(parent_rot)\n\t$(parent_com_to_joint)\n\t$(parent_com)")
        # println("child: $(child_pos),\n\t$(child_rot)\n\t$(child_com_to_joint)\n\t$(child_com)")
        # println()
        # TODO ask zac if this is a good way to do ori constraint for fixed
        ori_diff = parent_rpy - child_rpy

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
            cross_product = cross(parent_axis_world_frame, child_axis_world_frame)
            parent_axis_perp = orthogonal_complement(parent_frame_axis)
            ori_diff = parent_axis_perp * parent_rot' * cross_product
        elseif joint.joint_type isa RBD.Fixed
        elseif joint.joint_type isa RBD.Prismatic
            parent_frame_axis = RBD.rotation(RBD.joint_to_predecessor(joint))' * joint.joint_type.axis
            parent_axis_world_frame = parent_rot * parent_frame_axis
            parent_axis_world_perp = orthogonal_complement(parent_axis_world_frame)
            pos_diff = parent_axis_world_perp * pos_diff
        elseif joint.joint_type isa RBD.QuaternionFloating
            # no constraint
            continue
        else
            throw(DomainError(string(joint.joint_type), "unsupported joint type"))
        end
        push!(C, cat(pos_diff, ori_diff, dims=1))
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
    mechanism = RBD.parse_urdf(filename, remove_fixed_tree_joints=false,
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
        ForwardDiff.jacobian(constraint, state) * attitude_jacobian_from_configs(state)
    end

    function lagrangian(q1, q2, λ, Δt)
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
        T = vel'*M*vel
        U = sum(body.inertia.mass * gravity * (q1i[3] + q2i[3])/2
                for (body, q1i, q2i) in zip(mass_bodies, q1_groups[mass_body_idxs], q2_groups[mass_body_idxs]))
        C = constraint(q2)
        Cλ = isempty(C) ? 0 : C'*λ  # λ ignored if no constraints
        Δt * (T - U + Cλ)
    end

    mechanism, mass_matrix, constraint, constraint_jac, lagrangian
end

function simulate_unconstrained_system(q1, q2, Δt, lagrangian, time)
    D1(first, second) = convert(Array{Float64}, BlockDiagonal([Matrix(I, 3, 3), attitude_jacobian(first[4:7])]))' * ForwardDiff.gradient(first -> lagrangian(first, second, [], Δt), first)
    D2(first, second) = convert(Array{Float64}, BlockDiagonal([Matrix(I, 3, 3), attitude_jacobian(second[4:7])]))' * ForwardDiff.gradient(second -> lagrangian(first, second, [], Δt), second)
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

function simulate_constrained_system(q1, q2, Δt, lagrangian, num_constraints, time)
    D1(first, second, λ) = attitude_jacobian_from_configs(first)' * ForwardDiff.gradient(first -> lagrangian(first, second, λ, Δt), first)
    D2(first, second, λ) = attitude_jacobian_from_configs(second)' * ForwardDiff.gradient(second -> lagrangian(first, second, λ, Δt), second)
    qs = [q1, q2]
    t = 2 * Δt  # first two configs given
    λ₁, λ₂ = zeros(num_constraints), zeros(num_constraints)
    while t < time
        # x contains q3 ∈ ℜˢ, λ₁ ∈ ℜᶜ, λ₂ ∈ ℜᶜ, where s = num_configs, c = num_constraints
        objective(x) = D2(q1, q2, x[length(q2)+1:length(q2)+num_constraints]) + D1(q2, x[1:length(q2)], x[length(q2)+1+num_constraints:end])
        x0 = cat(q2, λ₁, λ₂, dims=1)
        x = newton(objective, x0, quat_adjust=true, len_config=length(q2))
        q3 = x[1:length(q2)]
        λ₁ = x[length(q2)+1:length(q2)+num_constraints]
        λ₂ = x[length(q2)+1+num_constraints:end]
        push!(qs, q3)
        q1, q2 = q2, q3
        t += Δt
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


mechanism, mass_matrix, constraint_fn, jacobian_fn, lagrangian = process_urdf(acrobot_path, floating=false)
num_constraints = length(constraint_fn(get_test_state(mechanism)))

function get_acrobot_state(θ₁, θ₂)
    return [0, 0, 0, 1, 0, 0, 0,  # fixed
            -0.5sin(θ₁), 0.15, -0.5cos(θ₁), cos(0.5θ₁), 0, sin(0.5θ₁), 0,  # upper link
            -sin(θ₁)-sin(θ₁+θ₂), 0.25, -cos(θ₁)-cos(θ₁+θ₂), cos(0.5(θ₁+θ₂)), 0, sin(0.5(θ₁+θ₂)), 0]  # lower link
end

# for acrobot
test_state0 = get_acrobot_state(0, 0)
test_state1 = get_acrobot_state(0.05, 0.06)

# for singleton
# test_state0 = cat(zeros(3), [1, 0, 0, 0], dims=1)
# test_state1 = cat(zeros(3), [0.999, 0.0, 0.0436, 0.0001], dims=1)

test_state0, test_state1
get_test_state(mechanism)
constraint_fn(test_state1)
# lagrangian(test_state0, test_state1, [], 0.01)
# qs = simulate_unconstrained_system(test_state0, test_state1, 0.1, lagrangian, 3)
qs = simulate_constrained_system(test_state0, test_state1, 0.1, lagrangian, num_constraints, 2)
[norm(constraint_fn(q)) for q in qs]
vis = Visualizer()
delete!(vis)
render(vis)

# axis_case ∈ ["maximal", "intermediate", "minimal"]
axis_case = "maximal"
if axis_case == "maximal"
    # singleton.urdf should have Ixx = 1.2, Iyy = 2.4, Izz = 0.8
    # <inertia ixx="1.2" ixy="0" ixz="0" iyy="2.4" iyz="0" izz="0.8" />
    box_dims = [0.4, 0.2, 0.6]
elseif axis_case == "intermediate"
    # singleton.urdf should have Ixx = 2.4, Iyy = 1.2, Izz = 0.8
    # <inertia ixx="2.4" ixy="0" ixz="0" iyy="1.2" iyz="0" izz="0.8" />
    box_dims = [0.2, 0.4, 0.6]
elseif axis_case == "minimal"
    # singleton.urdf should have Ixx = 1.2, Iyy = 0.8, Izz = 2.4
    # <inertia ixx="1.2" ixy="0" ixz="0" iyy="0.8" iyz="0" izz="2.4" />
    box_dims = [0.4, 0.6, 0.2]
else
    throw(DomainError(string(axis_case), "unsupported axis case"))
end

# animate
setobject!(vis[:link], Rect(Vec(-box_dims./2...), Vec(box_dims...)))
anim = Animation()
for i = 1:length(qs)
    pos_i, quat_i = qs[i][1:3], Quaternion(qs[i][4:7]...)
    angle_i, axis_i = angleaxis(quat_i)
    atframe(anim, i) do
        settransform!(vis[:link], compose(Translation(qs[i][1:3]), LinearMap(AngleAxis(angle_i, axis_i...))))
    end
end
setanimation!(vis, anim)
