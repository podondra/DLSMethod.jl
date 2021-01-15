module DLSMethod

using LinearAlgebra

export dls_fit!

function fit(X::Matrix{Float64}, y::Vector{Float64})::Vector{Float64}
    return inv(X' * X) * X' * y
end

function fit!(
        X::Matrix{Float64}, y::Vector{Float64}, n::Int64, a::Vector{Float64})
    X_view = view(X, 1:n, :)
    y_view = view(y, 1:n)
    a[:] = inv(X_view' * X_view) * X_view' * y_view
    return nothing
end

function arrange_layer!(
        X::Matrix{Float64}, y::Vector{Float64}, a::Vector{Float64},
        n_current::Int64, width::Float64, removal_parameter::Float64)::Int64
    remove_distance = removal_parameter * width
    while true
        n_init = n_current
        i = 1
        while i <= n_current
            y_pred = dot(view(X, i, :), a)
            if abs(y[i] - y_pred) >= remove_distance
                y_pred = dot(view(X, n_current, :), a)
                if abs(y[n_current] - y_pred) < remove_distance
                    # swap
                    X[i, :], X[n_current, :] = X[n_current, :], X[i, :]
                    y[i], y[n_current] = y[n_current], y[i]
                end
                n_current -= 1
            else
                i += 1
            end
        end
        if (n_current == n_init) || (n_current <= length(a) + 3)
            break
        end
        fit!(X, y, n_current, a)
    end
    return n_current
end

function dls_fit!(
        X::Matrix{Float64}, y::Vector{Float64},
        k::Int64, removal_parameter::Float64)::Tuple{Int64, Vector{Float64}}
    n, m = size(X)
    subset_n, subset_dls = Int64[], Float64[]
    a = Vector{Float64}(undef, m)
    fit!(X, y, n, a)

    while n > m + 3
        d = abs.(view(y, 1:n) - view(X ,1:n, :) * a)
        width = maximum(d)
        dls = sum(d .^ 2) / width .^ k

        push!(subset_n, n)
        push!(subset_dls, dls)

        n = arrange_layer!(X, y, a, n, width, removal_parameter)
    end

    n_best = subset_n[argmax(subset_dls)]
    fit!(X, y, n_best, a)
    return n_best, a
end

end # module
