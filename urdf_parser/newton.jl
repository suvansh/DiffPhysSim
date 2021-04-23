include("utils.jl")
function φ(ang)
	cat(1, ang, dims=1) ./ √(1+norm(ang)^2)
end

"""
	Newton; from https://github.com/thowell/motion_planning/blob/b84f6c729bd7a6e24e407ed27b6bf1a77becefe6/src/solvers/newton.jl
"""
function newton(res::Function, x;
		tol_r = 1.0e-8, tol_d = 1.0e-6, quat_adjust = false)
	y = copy(x)
	Δy = copy(x)

    r = res(y)

    iter = 0

	num_iters = 5
	num_iters_ls = 10

    while norm(r, 2) > tol_r && iter < num_iters
        ∇r = ForwardDiff.jacobian(res, y)
		if quat_adjust
			∇r *= convert(Array{Float64}, BlockDiagonal([Matrix(I, 3, 3), quat_ang_mat(y[4:7])]))
		end
		try
        	Δy = -1.0 * ∇r \ r
		catch
			@warn "implicit-function failure"
			return y
		end

        α = 1.0

		iter_ls = 0
        while α > 1.0e-8 && iter_ls < num_iters_ls
            if quat_adjust
				#ŷ = cat(y[1:3] + α * Δy[1:3], quat_L(y[4:7]) * φ(α * Δy[4:6]), dims=1)
				ŷ = y + α * convert(Array{Float64}, BlockDiagonal([Matrix(I, 3, 3), quat_ang_mat(y[4:7])])) * Δy
			else
				ŷ = y + α * Δy
			end
            r̂ = res(ŷ)

            if norm(r̂) < norm(r)
                y = ŷ
                r = r̂
                break
            else
                α *= 0.5
				iter_ls += 1
            end

			iter_ls == num_iters_ls && (@warn "line search failed")
        end

        iter += 1
    end

	iter == num_iters && (@warn "Newton failure")

    return y
end
