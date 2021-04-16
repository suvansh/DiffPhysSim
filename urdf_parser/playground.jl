using RigidBodyDynamics
RBD = RigidBodyDynamics
using LinearAlgebra
using BlockDiagonals
using DataStructures
using LightXML
using StaticArrays
using ForwardDiff
include("utils.jl")
include("structs.jl")

urdf_path = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/cassie-old.urdf"

mechanism = RBD.parse_urdf(urdf_path)
for body in RBD.bodies(mechanism)
    if RBD.has_defined_inertia(body)
        println(body.inertia.mass)
    end
end

body_order = Dict(Iterators.Enumerate(RBD.bodies(mechanism)))
links = [b.name for b in RBD.bodies(mechanism)]
joints = [j.name for j in RBD.joints(mechanism)]
"pair01_bar_a" ∈ names
println(RBD.joint_to_predecessor(RBD.joints(mechanism)[2]).mat)
println(RBD.bodies(mechanism)[3].inertia.cross_part)
println(RBD.joints(mechanism)[2].joint_type.axis)

Dict(b.id => i for (i, b) in Iterators.Enumerate(RBD.bodies(mechanism)))




xdoc = parse_file(urdf_path)
xroot = LightXML.root(xdoc)
@assert LightXML.name(xroot) == "robot"

xml_link_names = [attribute(item, "name") for item in get_elements_by_tagname(xroot, "link")]
xml_joint_names = [attribute(item, "name") for item in get_elements_by_tagname(xroot, "joint")]
"pair01_bar_a" ∈ xml_link_names
setdiff(Set(xml_link_names), Set(names))
