# ---------------------------------------------------------------------------
# Certified greedy reduced basis (eigenvector continuation) -- MODEL AGNOSTIC.
#
# Everything below knows only three things:
#   * the parameter box,
#   * the AFFINE pieces of the Hamiltonian, as callables H_j(x) -> H_j * x,
#   * a solver returning the k lowest exact eigenpairs at a parameter point.
# Nothing here mentions Ising, O(2), a sector, or a particular ED backend, so
# the same code drives this package's coupled-basis solver and FuzzifiED alike.
#
# The construction rests on
#     H(θ) = H_1 + θ_1 H_2 + ... + θ_p H_{p+1},
# affine in the couplings.  With an orthonormal snapshot basis Q the reduced
# operators A_j = Q'H_j Q and the blocks B_{jl} = (H_j Q)'(H_l Q) are formed
# ONCE, after which for ANY θ both the reduced eigenpair and the TRUE residual
# norm follow in O(r^2), with no access to the full space:
#
#     || (H(θ) - E) Q y ||^2  =  y' B(θ) y - E^2       (Q'Q = I, |y| = 1)
#
# That is a rigorous a-posteriori bound, not an estimate.  The greedy uses it
# to place the next snapshot where the bound is worst.
# ---------------------------------------------------------------------------

"""
    AffineOperator(pieces, dim)

The affine decomposition of a parametric Hamiltonian.  `pieces[j](x)` must
return `H_j * x`; the Hamiltonian at parameter `θ` (length `p = length(pieces)-1`)
is `H_1 + sum_j θ_j H_{j+1}`.  `dim` is the dimension of the vector space.

Any backend that can apply its Hamiltonian pieces to a vector can be wrapped:
see [`jscheme_ising_affine`](@ref) and the FuzzifiED adapter in
`examples/tutorial.jl`.
"""
struct AffineOperator{F}
    pieces::Vector{F}
    dim::Int
end
nparams(H::AffineOperator) = length(H.pieces) - 1
coeffs(H::AffineOperator, θ) = (1.0, float.(Tuple(θ))...)

"""
    Emulator

Reduced model: `A[j]` the projected affine pieces, `k` levels tracked, `pts` the
snapshot parameters the greedy chose, `dim` the full dimension, `res` the final
certified residual bound.
"""
struct Emulator
    A::Vector{Matrix{Float64}}
    k::Int
    pts::Vector{Vector{Float64}}
    dim::Int
    res::Float64
end

"""
    emulate(em, θ; k = em.k)

The `k` lowest Ritz values at `θ`.  Cost is O(r^3) in the basis size, wholly
independent of the full dimension.
"""
function emulate(em::Emulator, θ; k::Int = em.k)
    c = (1.0, float.(Tuple(θ))...)
    M = sum(c[j] .* em.A[j] for j in eachindex(em.A))
    eigen(Symmetric(M)).values[1:k]
end

"""
    greedy(H::AffineOperator, solve, box; k, tol, maxsnap, ncand, verbose)

Build an [`Emulator`](@ref) over `box`, a vector of `(lo, hi)` pairs, one per
parameter.  `solve(θ, k)` must return `(values, vectors)` for the `k` lowest
exact eigenpairs at `θ`, with vectors as columns in the same basis the
`AffineOperator` acts on.

Snapshots are placed greedily at the worst certified residual over a tensor
grid of `ncand` points per parameter.
"""
function greedy(H::AffineOperator, solve, box; k::Int = 1, tol = 1e-8,
                maxsnap = 60, ncand = 21, verbose = false)
    p = nparams(H); dim = H.dim; np1 = p + 1
    @assert length(box) == p "box has $(length(box)) ranges but H has $p parameters"
    axes_ = [collect(range(lo, hi; length = ncand)) for (lo, hi) in box]
    cands = vec([collect(t) for t in Iterators.product(axes_...)])
    θ = [(lo + hi) / 2 for (lo, hi) in box]
    ec = solve(θ, k)[1]
    σ = k == 1 ? ec[1] : (ec[1] + ec[k]) / 2       # shift kills the E^2 floor
    Q = zeros(dim, 0); W = [zeros(dim, 0) for _ in 1:np1]
    pts = Vector{Float64}[]; res = NaN
    for _ in 1:maxsnap
        V = solve(θ, k)[2]
        added = 0
        for j in 1:min(k, size(V, 2))
            v = Vector{Float64}(real.(V[:, j]))
            v .-= Q * (Q' * v); n1 = norm(v)
            v .-= Q * (Q' * v); nv = norm(v)
            (nv < 1e-6 || nv < 0.5 * n1) && continue
            v ./= nv; Q = hcat(Q, v); added += 1
            for j2 in 1:np1
                w = H.pieces[j2](v); j2 == 1 && (w = w .- σ .* v)
                W[j2] = hcat(W[j2], w)
            end
        end
        added == 0 && break
        push!(pts, copy(θ))
        A = [Q' * W[j] for j in 1:np1]
        B = [W[a]' * W[b] for a in 1:np1, b in 1:np1]
        worst, θn = -1.0, θ
        for c in cands
            cc = (1.0, c...)
            F = eigen(Symmetric(sum(cc[j] .* A[j] for j in 1:np1)))
            Bm = sum(cc[a] * cc[b] .* B[a, b] for a in 1:np1, b in 1:np1)
            for j in 1:min(k, length(F.values))
                y = F.vectors[:, j]
                r2 = max(dot(y, Bm * y) - F.values[j]^2, 0.0)
                r2 > worst && (worst = r2; θn = c)
            end
        end
        res = sqrt(worst)
        verbose && @printf("    r=%2d  certified residual %.2e\n", size(Q, 2), res)
        res < tol && break
        θ = θn
    end
    orth = maximum(abs.(Q' * Q - I))
    orth > 1e-10 && error("snapshot basis lost orthogonality ($orth)")
    Wr = [reduce(hcat, [H.pieces[j](Q[:, i]) for i in 1:size(Q, 2)]) for j in 1:np1]
    # undo the shift: store the UNSHIFTED reduced operators
    Emulator([Q' * Wr[j] for j in 1:np1], k, pts, dim, res)
end
