using Rotations, StaticArrays
using LightXML
using BlockDiagonals


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

function world_attitude_jacobian(quat)
    w, x, y, z = quat
	0.5 * [
        -x -y -z;
         w  z -y;
        -z  w  x;
         y -x  w
    ]
end

function body_attitude_jacobian(quat)
    w, x, y, z = quat
	# from planning with attitude paper
    0.5 * [
        -x -y -z;
         w -z  y;
         z  w -x;
        -y  x  w
    ]
end

function world_attitude_jacobian_from_configs(q)
    convert(Array{Float64}, BlockDiagonal([
        BlockDiagonal([Matrix(I, 3, 3), world_attitude_jacobian(chunk[4:7])])
            for chunk in Iterators.partition(q, 7)
    ]))
end

function body_attitude_jacobian_from_configs(q)
    convert(Array{Float64}, BlockDiagonal([
        BlockDiagonal([Matrix(I, 3, 3), body_attitude_jacobian(chunk[4:7])])
            for chunk in Iterators.partition(q, 7)
    ]))
end

function config_diff(q1, q2)
	""" q1-q2, where each consists of 7-element chunks of position and quat """
	diff_arr = []
	for (q1i, q2i) in zip(Iterators.partition(q1, 7), Iterators.partition(q2, 7))
		pos_diff = q1i[1:3] - q2i[1:3]
		quat_diff = -L(q2i[4:7])' * q1i[4:7]
		diff = cat(pos_diff, quat_diff, dims=1)
		push!(diff_arr, diff)
	end
	cat(diff_arr..., dims=1)
end

function φ(ang)
	cat(1, ang, dims=1) ./ √(1+norm(ang)^2)
end

function φ_vec(ang)
	cat(√(1-norm(ang)^2), ang, dims=1)
end

function geodesic_quat_dist(q1, q2)
	return min(1 - q1 ⋅ q2, 1 + q1 ⋅ q2)
end

function ϕ(quat)
	w, x, y, z = quat
	[x, y, z] ./ w
end


function L(quat)
    w, x, y, z = quat
    [
        w  -x  -y  -z;
        x   w  -z   y;
        y   z   w  -x;
        z  -y   x   w;
    ]
end

function R(quat)
    w, x, y, z = quat
    [
        w  -x  -y  -z;
        x   w   z  -y;
        y  -z   w   x;
        z   y  -x   w;
    ]
end

H = @SMatrix [0.0 0 0; 1 0 0; 0 1 0; 0 0 1]
S = @SVector[1, 0, 0, 0]
T = @SMatrix [1.0 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 -1]

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

function resolve_constraint(q₀, C, J, num_constraints; maxiters=40, α=1e-4, tol=1e-12, ls=false)
    λ = zeros(num_constraints)
    iter = 0
    config_dim = length(q₀) * 6 ÷ 7  # since we drop a dimension from the quat
    q = copy(q₀)
    b = [world_attitude_jacobian_from_configs(q)' * config_diff(q₀, q); -C(q)]  # from utils.jl
    while iter < maxiters
        iter += 1
        jac = J(q)
        A = [Matrix(I, config_dim, config_dim)              jac';
            jac             -α*Matrix(I, num_constraints, num_constraints)]
        res = A \ b  # TODO confirm that the intermediate λs don't matter?
        Δq, λ = res[1:config_dim], res[config_dim+1:end]

        if ls
			β = 1
            while β > 1e-8
                q_arr = []
                for (qi, dqi) in zip(Iterators.partition(q, 7), Iterators.partition(Δq, 6))
                    pos = qi[1:3] + β * dqi[1:3]
                    rot = L(qi[4:7]) * φ(β * dqi[4:6])  # φ_vec from utils.jl
                    push!(q_arr, cat(pos, rot, dims=1))
                end
                q̂ = cat(q_arr..., dims=1)
                b̂ = [world_attitude_jacobian_from_configs(q̂)' * config_diff(q₀, q̂); -C(q̂)]
                if maximum(abs, b̂) < maximum(abs, b)
                    q = q̂
                    b = b̂
                    break
                end
                β *= 0.5
            end
        else
			β = 0.5
            q_arr = []
            for (qi, dqi) in zip(Iterators.partition(q, 7), Iterators.partition(Δq, 6))
                pos = qi[1:3] + β * dqi[1:3]
                rot = L(qi[4:7]) * φ(β * dqi[4:6])  # φ_vec from utils.jl
                push!(q_arr, cat(pos, rot, dims=1))
            end
            q = cat(q_arr..., dims=1)
            b = [world_attitude_jacobian_from_configs(q)' * config_diff(q₀, q); -C(q)]
        end
		println(maximum(abs, Δq))
        if maximum(abs, Δq) < tol
            break
        end
    end
    if iter == maxiters
        @warn "didn't converge"
    end
    q
end
