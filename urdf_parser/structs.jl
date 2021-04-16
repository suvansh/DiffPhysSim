using LinearAlgebra
using LightXML
include("utils.jl")

baremodule JointTypeEnums
    using Base: @enum
    @enum JointTypeEnum revolute continuous prismatic fixed floating planar
end

total_constraints_dict = Dict(
    JointTypeEnums.fixed => 0,
    JointTypeEnums.planar => 3,
    JointTypeEnums.revolute => 5,
    JointTypeEnums.continuous => 5,
    JointTypeEnums.prismatic => 5,
    JointTypeEnums.fixed => 6
)

struct JointType
    type::JointTypeEnums.JointTypeEnum
    num_constraints::UInt8
    JointType(type) = new(type, total_constraints_dict[type])
end

struct JointLimit
    lower::Float64
    upper::Float64
    effort::Float64
    velocity::Float64
end

mutable struct Link
    state_idx::UInt8  # idx into the global state where this link's state begins
    name::String
    mass::Float64
    inertia::SMatrix{3,3,Float64}
    has_defined_inertia::Bool
    pos_offset::SVector{3,Float64}
    rot_offset::SMatrix{3,3,Float64}

    function Link(state_idx::Number, obj::XMLElement)
        L = new()
        inertial = find_element(obj, "inertial")
        origin = find_element_default(inertial, "origin", nothing)
        L.state_idx = state_idx
        L.name = attribute(obj, "name")
        L.pos_offset = parse_vector(Float64, origin, "xyz", "0 0 0")
        L.rot_offset = rpy_to_rot(parse_vector(Float64, origin, "rpy", "0 0 0"))
        L
    end
end


abstract type AbstractJointLink end


mutable struct Joint{T<:AbstractJointLink}
    name::String
    type::JointType
    parent::T
    child::T
    limit::JointLimit

    function Joint{T}(obj::XMLElement, name_to_link::Dict) where {T<:AbstractJointLink}
        j = new{T}()
        # TODO look into how origin relates to offset (child or parent, pos or neg)
        # and how to get offset of other link (parent or child)
        axis = parse_vector(Float64, find_element(obj, "axis"), "xyz", "1 0 0")
        origin_xml = find_element(obj, "origin")
        parent_link = name_to_link[attribute(find_element(obj, "parent"), "link")]
        # NOTE 4th arg is CoG --> anchor, assumption that state pos gives CoG pos
        j.parent = T(parent_link,
                        j,
                        axis,
                        parse_vector(Float64, origin_xml, "xyz", "0 0 0") - parent_link.pos_offset,
                        rpy_to_rot(parse_vector(Float64, origin_xml, "rpy", "0 0 0")) * parent_link.rot_offset')
        child_link = name_to_link[attribute(find_element(obj, "child"), "link")]
        j.child = T(child_link,
                    j,
                    child_link.rot_offset' * axis,
                    -child_link.pos_offset,
                    child_link.rot_offset')
        j.name = attribute(obj, "name")
        j.type = JointType(getproperty(JointTypeEnums, Symbol(attribute(obj, "type"))))
        joint_limit = find_element(obj, "limit")
        if !isnothing(joint_limit)
            lower = attribute(joint_limit, "lower")
            upper = attribute(joint_limit, "upper")
            effort = attribute(joint_limit, "effort")
            velocity = attribute(joint_limit, "velocity")
            j.limit = JointLimit(isnothing(lower) ? 0 : lower,
                                    isnothing(upper) ? 0 : upper,
                                    effort,
                                    velocity)
        end
        return j
    end
end


mutable struct JointLink <: AbstractJointLink
    link::Link
    joint::Joint
    axis::SVector{3,Float64}  # body frame joint axis
    pos_offset::SVector{3,Float64}  # body frame offset from link CoG to anchor
    rot_offset::SMatrix{3,3,Float64}  # rot matrix from body frame to parent frame
end

Joint(obj::XMLElement, name_to_link::Dict) = Joint{JointLink}(obj, name_to_link)
