using Rotations, StaticArrays


function hat(vector)
    @assert length(vector) == 3
    [0 -vector[3] vector[2];
    vector[3] 0 -vector[1];
    -vector[2] vector[1] 0]
end

function find_element_default(e::Union{XMLElement, Nothing}, name::String, default::Any)
    usedefault = e == nothing || find_element(e, name) == nothing
    usedefault ? default : find_element(e, name)
end

function parse_vector(::Type{T}, e::Union{XMLElement, Nothing}, name::String, default::String) where {T}
    # taken from RigidBodyDynamics
    usedefault = e == nothing || attribute(e, name) == nothing
    [parse(T, str) for str in split(usedefault ? default : attribute(e, name))]
end

function parse_scalar(::Type{T}, e::Union{XMLElement, Nothing}, name::String, default::String) where {T}
    # taken from RigidBodyDynamics
    usedefault = e == nothing || attribute(e, name) == nothing
    parse(T, usedefault ? default : attribute(e, name))
end

function orthogonal_complement(vec)
    """ returns 2x3 with the two components of the orthogonal complement """
    @assert length(vec) == 3
    qr([vec I]).Q[:,2:3]'
end

function attitude_jacobian(quat)
    w, x, y, z = quat
    [
        -x -y -z;
         w  z -y;
        -z  w  x;
         y -x  w
    ]
end

function attitude_jacobian_from_configs(q)
    # println(size(q))
    convert(Array{Float64}, BlockDiagonal([
        BlockDiagonal([Matrix(I, 3, 3), attitude_jacobian(chunk[4:7])])
            for chunk in Iterators.partition(q, 7)
    ]))
end

function quat_L(quat)
    w, x, y, z = quat
    [
        w  -x  -y  -z;
        x   w  -z   y;
        y   z   w  -x;
        z  -y   x   w;
    ]
end

function quat_R(quat)
    w, x, y, z = quat
    [
        w  -x  -y  -z;
        x   w   z  -y;
        y  -z   w   x;
        z   y  -x   w;
    ]
end

function quat_to_rot(quat)
    @assert length(quat) == 4
    # assumes quaternion has scalar part first
    q0, q1, q2, q3 = quat
    @SMatrix [
        2*q0^2+2*q1^2-1     2*q1*q2-2*q0*q3     2*q1*q3+2*q0*q2;
        2*q1*q2+2*q0*q3     2*q0^2+2*q2^2-1     2*q2*q3-2*q0*q1;
        2*q1*q3-2*q0*q2     2*q2*q3+2*q0*q1     2*q0^2+2*q3^2-1
    ]
end

function quat_to_rpy(quat)
    @assert length(quat) == 4
    q0, q1, q2, q3 = quat
    ϕ = atan(q2*q3 + q0*q1, 0.5 - (q1^2 + q2^2))
    θ = asin(-2 * (q1*q3 - q0*q2))
    ψ = atan(q1*q2 + q0*q3, 0.5 - (q2^2 + q3^2))
    @SVector [ψ, θ, ϕ]
end

function rpy_to_rot(rpy)
    @assert length(rpy) == 3
    r, p, y = rpy
    @SMatrix [
        cos(p)*cos(y)  -cos(r)*sin(y)+sin(r)*sin(p)*cos(y)  sin(r)*sin(y)+cos(r)*sin(p)*cos(y) ;
        cos(p)*sin(y)  cos(r)*cos(y)+sin(r)*sin(p)*sin(y)   -sin(r)*cos(y)+cos(r)*sin(p)*sin(y) ;
        -sin(p)        sin(r)*cos(p)                        cos(r)*cos(p)
    ]
end
