using ForwardDiff, Roots

function simulate_unconstrained_system(q1, q2, Δt, lagrangian, time)
    D1(first, second) = ForwardDiff.gradient(first => lagrangian(first, second, [], Δt), first)
    D2(first, second) = ForwardDiff.gradient(second => lagrangian(first, second, [], Δt), second)
    # solve D2(q1, q2) + D1(q2, q3) = 0 for q3
    # loop below
    qs = [q1, q2]
    t = 0
    while t < time
        const_term = D2(q1, q2)
        objective(q3) = const_term + D1(q2, q3)
        q3 = find_zero((objective, q3 -> ForwardDiff.jacobian(objective, q3)),
                        q2, Roots.Newton())
        push!(qs, q3)
        q1, q2 = q2, q3
        t += Δt
    end
    qs
end
