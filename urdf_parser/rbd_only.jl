import RigidBodyDynamics
using LinearAlgebra
RBD = RigidBodyDynamics
using LightXML
using StaticArrays
using MeshCatMechanisms
using MeshCat


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
        body1 = RBD.findbody(mechanism, attribute(xml_link1, "link"))
        H1 = RBD.Transform3D(RBD.frame_before(joint), RBD.default_frame(body1),
            RBD.parse_pose(T, xml_link1)...)
        xml_link2 = find_element(xml_loop, "link2")
        body2 = RBD.findbody(mechanism, attribute(xml_link2, "link"))
        H2 = RBD.Transform3D(RBD.frame_after(joint), RBD.default_frame(body2),
            RBD.parse_pose(T, xml_link2)...)
        RBD.attach!(mechanism, body1, body2, joint,
            joint_pose = H1,
            successor_pose = inv(H2))
    end
end

# TODO try with floating base and the puppet strings chopped off
# change the pi pi/2 angles to be the exact same
# path = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/urdfs/scissor_2d.urdf"
path = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/urdfs/bsm_2thread_nohanger.urdf"
# path = "/Users/sanjeev/GoogleDrive/CMU/Research/ConstrainedDynamics.jl/examples/examples_files/fourbar.urdf"
mechanism = RBD.parse_urdf(path, remove_fixed_tree_joints=false, floating=true, gravity=[0,0,0])
add_loop_joints!(mechanism, path)

state = RBD.MechanismState(mechanism)

joints = RBD.joints(mechanism)
result = RBD.DynamicsResult(mechanism)
jac = RBD.constraint_jacobian!(result, state)
svd(jac)

function torques!(torques::AbstractVector, t, state::RBD.MechanismState)
    torques[1:3] .= 0.001
end


ts, qs, vs = RBD.simulate(state, 6., Δt=0.01, torques!)

animation = Animation(mvis, ts, qs)
setanimation!(mvis, animation)


vis = Visualizer()
render(vis)
delete!(vis)
mvis = MechanismVisualizer(mechanism, URDFVisuals(path), vis)
