import RigidBodyDynamics
RBD = RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using LinearAlgebra
using BlockDiagonals
using DataStructures
using LightXML
using StaticArrays
using ForwardDiff
include("utils.jl")
include("structs.jl")

urdf_path = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/acrobot.urdf"
# link_to_id = OrderedDict(link => i for (i, link) in enumerate(xml_links))
# id_to_link = OrderedDict(i => link for (link, i) in link_to_id)
# joint_to_id = OrderedDict(joint => i for (i, joint) in enumerate(xml_joints))
# id_to_joint = OrderedDict(i => joint for (joint, i) in joint_to_id)
# joint_to_name = OrderedDict(joint => attribute(joint, "name") for joint in xml_joints)
# name_to_joint = OrderedDict(name => joint for (joint, name) in joint_to_name)
# link_to_name = OrderedDict(link => attribute(link, "name") for link in xml_links)
# name_to_link = OrderedDict(name => link for (link, name) in link_to_name)
# joint_to_links = OrderedDict(joint_to_id[xml_joint]
#                     => [link_to_id[name_to_link[attribute(find_element(xml_joint, "parent"), "link")]],
#                         link_to_id[name_to_link[attribute(find_element(xml_joint, "child"), "link")]]]
#                     for xml_joint in xml_joints)
# parent_id, child_id = joint_to_links[1]


"""
need a function that takes in state and link ID and returns the link-specific state
constraint function:
    take in state
    loop over joints
        for parent and child JointLinks
            get link-state from state and JointLink.link
            get axis in body frame from JointLink.axis
            get anchor offset from CoG from JointLink.offset
            using axis and link-state's orientation, get axis in world frame
            using offset and link-state's position and orientation, get offset in world frame
"""



# function process_joint_jac_old!(jac, state, jac_idx, parent_id, child_id, xml_joint, joint_type)
#     parent_start_idx = 6 * (parent_id - 1) + 1
#     child_start_idx = 6 * (child_id - 1) + 1
#
#     if joint_type == "continuous"
#         """ position-linear Jacobian """
#         jac[jac_idx:jac_idx+2, parent_start_idx:parent_start_idx+2] = Matrix(I, 3, 3)
#         jac[jac_idx:jac_idx+2, child_start_idx:child_start_idx+2] = -Matrix(I, 3, 3)
#         """ position-angular Jacobian """
#         # d/dt(R(q)a) = -sum(a_i * \hat{R(q)_i}) * ω
#         # where a is joint anchor, R(q) is rotation matrix associated with state
#         # TODO deal with rpy attribute of origins
#         anchor_parent = parse_vector(Float64, find_element(xml_joint, "origin"),
#                             "xyz", "0 0 0")
#         # TODO check: negative for child because origin specifies shift of new origin from anchor?
#         anchor_child = -parse_vector(Float64,
#                             find_element(
#                                 find_element(id_to_link[child_id], "visual"),
#                                 "origin"),
#                             "xyz", "0 0 0")
#         # TODO get state for each link assoc with joint
#         # TODO figure out idxs again
#         parent_state = nothing
#         child_state = nothing
#         rot_parent = rot_matrix(parent_state)
#         rot_child = rot_matrix(child_state)
#         jac[jac_idx:jac_idx+2, parent_start_idx+3:parent_start_idx+5] = hat(rot_parent * anchor_parent)
#         jac[jac_idx:jac_idx+2, child_start_idx+3:child_start_idx+5] = hat(rot_child * anchor_child)
#         """ orientation-angular Jacobian """
#         axis = parse_vector(Float64, find_element(xml_joint, "axis"), "xyz", "1 0 0")
#         # 2x3 representing 2 orth basis vecs of the space orth to the joint axis
#         axis_perp = qr([axis I]).Q[:,2:3]'
#         jac[jac_idx+3:jac_idx+4, parent_start_idx+3:parent_start_idx+5] = axis_perp
#         jac[jac_idx+3:jac_idx+4, child_start_idx+3:child_start_idx+5] = -axis_perp
#     else
#         throw(DomainError(joint_type, "unsupported joint type"))
#     end
# end
#
# function process_joint_jac!(jac, state, joint, jac_idx)#, parent_id, child_id, xml_joint, joint_type)
#     parent = joint.parent.link
#     child = joint.child.link
#
#     if joint.type == JointTypeEnum.continuous
#         """ position-linear Jacobian """
#         jac[jac_idx:jac_idx+2, parent.state_idx:parent.state_idx+2] = Matrix(I, 3, 3)
#         jac[jac_idx:jac_idx+2, child.state_idx:child.state_idx+2] = -Matrix(I, 3, 3)
#         """ position-angular Jacobian """
#         # d/dt(R(q)a) = -hat(R(q)a) * ω
#         # where a is joint anchor, R(q) is rotation matrix associated with state
#         # TODO deal with rpy attribute of origins
#         anchor_parent = parse_vector(Float64, find_element(xml_joint, "origin"),
#                             "xyz", "0 0 0")
#         # TODO check: negative for child because origin specifies shift of new origin from anchor?
#         anchor_child = -parse_vector(Float64,
#                             find_element(
#                                 find_element(id_to_link[child_id], "visual"),
#                                 "origin"),
#                             "xyz", "0 0 0")
#         # TODO figure out idxs again
#         parent_state = state[parent.state_idx:parent.state_idx+5]
#         child_state = state[child.state_idx:child.state_idx+5]
#         rot_parent = quat_to_rot(parent_state[4:7])
#         rot_child = quat_to_rot(child_state[4:7])
#         jac[jac_idx:jac_idx+2, parent.state_idx+3:parent.state_idx+5] = hat(rot_parent * anchor_parent)
#         jac[jac_idx:jac_idx+2, child.state_idx+3:child.state_idx+5] = hat(rot_child * anchor_child)
#         """ orientation-angular Jacobian """
#         # TODO axis
#         axis = parse_vector(Float64, find_element(xml_joint, "axis"), "xyz", "1 0 0")
#         # 2x3 representing 2 orth basis vecs of the space orth to the joint axis
#         axis_perp = qr([axis I]).Q[:,2:3]'
#         jac[jac_idx+3:jac_idx+4, parent.state_idx+3:parent.state_idx+5] = axis_perp
#         jac[jac_idx+3:jac_idx+4, child.state_idx+3:child.state_idx+5] = -axis_perp
#     else
#         throw(DomainError(joint_type, "unsupported joint type"))
#     end
# end


function constraint_vector(state, joints)
    C = []
    for joint in joints
        parent_state = state[joint.parent.link.state_idx:joint.parent.link.state_idx+6]
        child_state = state[joint.child.link.state_idx:joint.child.link.state_idx+6]
        if joint.type.type == JointTypeEnums.continuous
            """ positional constraint """
            parent_rot = quat_to_rot(parent_state[4:7])
            parent_pos = parent_state[1:3]
            child_rot = quat_to_rot(child_state[4:7])
            child_pos = child_state[1:3]
            parent_anc_wf = parent_pos + parent_rot * joint.parent.pos_offset
            child_anc_wf = child_pos + child_rot * joint.child.pos_offset
            println(parent_pos, joint.parent.pos_offset, parent_rot)
            println(child_pos, joint.child.pos_offset, child_rot)
            println(parent_anc_wf, child_anc_wf)
            pos_diff = parent_anc_wf - child_anc_wf

            """ orientation constraint """
            # we want a minimal (2d) representation of the 3d axis error
            # take cross product of the world frame axes and project onto
            # orthogonal complement of parent axis, finding coords in this plane
            parent_axis = parent_rot * joint.parent.axis
            child_axis = child_rot * joint.child.axis
            cross_product = cross(parent_axis, child_axis)
            parent_axis_perp = qr([joint.parent.axis I]).Q[:,2:3]'
            ori_diff = parent_axis_perp * parent_rot' * cross_product

            push!(C, cat(pos_diff, ori_diff, dims=1))
        else
            throw(DomainError(string(joint.type.type), "unsupported joint type"))
        end
    end
    cat(C..., dims=1)
end


# XML parsing
function process_urdf(filename::String)
    # to get constraint jacobian, we process joints separately
    # since mechanism doesn't seem to store joint-to-link mapping
    xdoc = parse_file(filename)
    xroot = LightXML.root(xdoc)
    @assert LightXML.name(xroot) == "robot"

    xml_links = get_elements_by_tagname(xroot, "link")
    xml_joints = get_elements_by_tagname(xroot, "joint")
    # process links
    state_idx = 1
    name_to_link = Dict()
    link_list = []
    for xml_link in xml_links
        link = Link(state_idx, xml_link)
        push!(link_list, link)
        name_to_link[link.name] = link
        state_idx += 7
    end

    # mass matrix
    mechanism = RBD.parse_urdf(filename)
    for (body, link) in zip(RBD.bodies(mechanism), link_list)
        link.has_defined_inertia = RBD.has_defined_inertia(body)
        if link.has_defined_inertia
            link.mass = body.inertia.mass
            link.inertia = body.inertia.moment
        end
    end
    masses_matrix = BlockDiagonal(
        [BlockDiagonal([Matrix(link.mass*I, 3, 3), link.inertia])
            for link in link_list if (link.has_defined_inertia
                && link.mass != 0 && norm(link.inertia) != 0)]
    )

    # process joints
    joint_list = []
    for xml_joint in xml_joints
        push!(joint_list, Joint(xml_joint, name_to_link))
    end

    function constraint_fn(state)
        constraint_vector(state, joint_list)
    end
    function jacobian_fn(state)
        # to get the second dim from 7*num_links to 6*num_links
        # TODO this is prob a slow way to do this
        quat_ang_mat_full = BlockDiagonal([
            BlockDiagonal([Matrix(I, 3, 3), quat_ang_mat(chunk[4:7])])
                for chunk in Iterators.partition(state, 7)
        ])
        J = ForwardDiff.jacobian(constraint_fn, state)
        J * quat_ang_mat_full
    end
    masses_matrix, constraint_fn, jacobian_fn
end


masses_matrix, constraint_fn, jacobian_fn = process_urdf(urdf_path)
# TODO test rotation constraint
test_state = [0, 0, 0, 1, 0, 0, 0,
                0, 0.15, -0.5, 1, 0, 0, 0,
                0, 0.25, -2, 1, 0, 0, 0]
test_con = constraint_fn(test_state)
test_jac = jacobian_fn(test_state)
sub_jac = test_jac[:,7:12]


# plotting
vis = Visualizer()
render(vis)
delete!(vis)
mechanism = RBD.parse_urdf(urdf_path)
mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf_path), vis)
state = RBD.MechanismState(mechanism, randn(2), randn(2))


result = RBD.DynamicsResult{Float64}(mechanism)
cj = RBD.constraint_jacobian!(result, state)
result.constraintjacobian
RBD.num_velocities(mechanism)

# animation
t, q, v = simulate(state, 5.0)
animation = Animation(mvis, t, q)
setanimation!(mvis, animation)

fixed, shoulder, elbow = RBD.joints(mechanism)
world, base, upper, lower = RBD.bodies(mechanism)
body_dict = OrderedDict(body.id.value => body for body in bodies(mechanism))
joint_dict = OrderedDict(joint.id.value => joint for joint in joints(mechanism))

println(shoulder.joint_to_successor.x)
