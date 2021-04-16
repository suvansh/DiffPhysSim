"""
	Newton; from https://github.com/thowell/motion_planning/blob/b84f6c729bd7a6e24e407ed27b6bf1a77becefe6/src/solvers/newton.jl
"""
function newton(res::Function, x;
		tol_r = 1.0e-8, tol_d = 1.0e-6)
	y = copy(x)
	Δy = copy(x)

    r = res(y)

    iter = 0

	num_iters = 1000
	num_iters_ls = 50

    while norm(r, 2) > tol_r && iter < num_iters
        ∇r = ForwardDiff.jacobian(res, y)
		try
        	Δy = -1.0 * ∇r \ r
		catch
			@warn "implicit-function failure"
			return y
		end

        α = 1.0

		iter_ls = 0
        while α > 1.0e-8 && iter_ls < num_iters_ls
            ŷ = y + α * Δy
            r̂ = res(ŷ)

            if norm(r̂) < norm(r)
                y = ŷ
                r = r̂
				println(α)
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
