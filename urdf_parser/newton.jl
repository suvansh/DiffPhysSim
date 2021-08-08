using ForwardDiff
include("utils.jl")


"""
	Newton's method
"""
function newton(f::Function, j::Function, x; num_iters=10, tol=1.0e-14, len_config=0, ls_mult=0.5, merit_norm=2, print_jac=false)
	num_configs = len_config ÷ 7
	y = copy(x)
	r = f(y)
	iter = 0
	# while norm(r, merit_norm) > tol
	while maximum(abs, r) > tol
		∇r = j(y)
		Δy = -1.0 * ∇r \ r
		# line search
		α = 1
		while α > 1e-8
			yh_arr = []
			for (yi, dyi) in zip(Iterators.partition(y[1:len_config], 7), Iterators.partition(Δy[1:6*num_configs], 6))
				pos = yi[1:3] + α*dyi[1:3]
				# TODO here you could negate `rot`
				# use min(1+dot prod of quat, 1-dot prod) as quat dist metric
				# this φ should be diff for the non-MRP version. vec stays same, scalar is filled in to reach unit norm
				rot = L(yi[4:7]) * φ(α*dyi[4:6])
				# rot1 = L(yi[4:7]) * φ(α*dyi[4:6])
				# rot2 = L(yi[4:7]) * φ(-α*dyi[4:6])
				# if geodesic_quat_dist(yi[4:7], rot1) < geodesic_quat_dist(yi[4:7], rot2)
				# 	rot = rot1
				# else
				# 	rot = rot2
				# end

				push!(yh_arr, cat(pos, rot, dims=1))
			end
			ŷ = cat(yh_arr..., y[len_config+1:end] + α * Δy[6*num_configs+1:end], dims=1)
			r̂ = f(ŷ)
			if maximum(abs, r̂) < maximum(abs, r)
				y = ŷ
				r = r̂
				break
			else
				α *= ls_mult
			end
		end

		iter += 1
		if iter == num_iters
			@warn "Newton failed with norm $(maximum(abs, r))"
			break
		end
	end
	y
end
