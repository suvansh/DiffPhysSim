using ForwardDiff
include("utils.jl")


"""
	Newton's method
"""
function newton2(f::Function, x; num_iters=10, tol=1.0e-8, len_config=0, ls_mult=0.5, merit_norm=2)
	num_configs = len_config ÷ 7
	y = copy(x)
	r = f(y)
	iter = 0
	while norm(r, merit_norm) > tol
		∇r = ForwardDiff.jacobian(f, y)
		∇r = cat(∇r[:, 1:len_config] * world_attitude_jacobian_from_configs(y[1:len_config]),
					∇r[:,len_config+1:end], dims=2)
		Δy = -1.0 * ∇r \ r
		# line search
		α = 1.0
		while α > 1e-8
			yh_arr = []
			for (yi, dyi) in zip(Iterators.partition(y[1:len_config], 7), Iterators.partition(Δy[1:6*num_configs], 6))
				pos = yi[1:3] + α*dyi[1:3]
				rot = L(yi[4:7]) * φ(α*dyi[4:6])
				push!(yh_arr, cat(pos, rot, dims=1))
			end
			ŷ = cat(yh_arr..., y[len_config+1:end] + α * Δy[6*num_configs+1:end], dims=1)
			r̂ = f(ŷ)
			if norm(r̂, merit_norm) < norm(r, merit_norm)
				y = ŷ
				r = r̂
				break
			else
				α *= ls_mult
			end
		end
		# r = f(y)

		iter += 1
		if iter == num_iters
			@warn "Newton failed with norm $(norm(r, 2))"
			break
		end
	end
	y
end

function newton2_with_jac(f::Function, j::Function, x; apply_attitude=true, num_iters=10, tol=1.0e-8, len_config=0, ls_mult=0.5, merit_norm=2)
	num_configs = len_config ÷ 7
	y = copy(x)
	r = f(y)
	iter = 0
	while norm(r, merit_norm) > tol
		∇r = j(y)
		# println(size(∇r))
		if apply_attitude
			∇r = cat(∇r[:, 1:len_config] * world_attitude_jacobian_from_configs(y[1:len_config]),
					∇r[:,len_config+1:end], dims=2)
		end
		# println(size(∇r))
		# println("∇r: $(∇r), r: $(r)")
		Δy = -1.0 * ∇r \ r
		# line search
		α = 1.0
		while α > 1e-8
			yh_arr = []
			for (yi, dyi) in zip(Iterators.partition(y[1:len_config], 7), Iterators.partition(Δy[1:6*num_configs], 6))
				pos = yi[1:3] + α*dyi[1:3]
				rot = L(yi[4:7]) * φ(α*dyi[4:6])
				push!(yh_arr, cat(pos, rot, dims=1))
			end
			ŷ = cat(yh_arr..., y[len_config+1:end] + α * Δy[6*num_configs+1:end], dims=1)
			r̂ = f(ŷ)
			if norm(r̂, merit_norm) < norm(r, merit_norm)
				y = ŷ
				r = r̂
				break
			else
				α *= ls_mult
			end
		end
		# r = f(y)

		iter += 1
		if iter == num_iters
			@warn "Newton failed with norm $(norm(r, 2))"
			break
		end
	end
	y
end
