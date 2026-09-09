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
see [`jscheme_ising_affine`](@ref), which `examples/tutorial_ising.jl` uses.
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
    basis::Matrix{Float64}   # the orthonormal snapshots; observables need them
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
    emulate_state(em, θ; k = em.k)

As [`emulate`](@ref), but returning `(values, vectors)` of the reduced
eigenproblem.  The vectors are the coordinates `y` in the stored basis, which
is what an observable needs; `emulate` returns the energies alone.
"""
function emulate_state(em::Emulator, θ; k::Int = em.k)
    c = (1.0, float.(Tuple(θ))...)
    M = sum(c[j] .* em.A[j] for j in eachindex(em.A))
    F = eigen(Symmetric(M))
    return F.values[1:k], F.vectors[:, 1:k]
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
                maxsnap = 60, ncand = 21, verbose = false,
                monitor = nothing, monitor_tol = 1e-5)
    p = nparams(H); dim = H.dim; np1 = p + 1
    @assert length(box) == p "box has $(length(box)) ranges but H has $p parameters"
    axes_ = [collect(range(lo, hi; length = ncand)) for (lo, hi) in box]
    cands = vec([collect(t) for t in Iterators.product(axes_...)])
    θ = [(lo + hi) / 2 for (lo, hi) in box]
    ec = solve(θ, k, nothing)[1]
    σ = k == 1 ? ec[1] : (ec[1] + ec[k]) / 2       # shift kills the E^2 floor

    # Preallocate.  Growing these with hcat reallocates and copies the whole
    # dim x r block every snapshot -- O(dim r^2) of pure memory traffic, which
    # at dim ~ 1e6 dominates everything else in the loop.
    maxc = maxsnap * k
    Q = zeros(dim, maxc); W = [zeros(dim, maxc) for _ in 1:np1]
    r = 0
        pts = Vector{Float64}[]; res = NaN; guess = nothing
    monitor_prev = nothing; res_prev = NaN
    for _ in 1:maxsnap
        r_before = r
        V = solve(θ, k, guess)[2]
        added = 0
        for jj in 1:min(k, size(V, 2))
            r == maxc && break
            v = Vector{Float64}(real.(V[:, jj]))
            Qv = view(Q, :, 1:r)
            v .-= Qv * (Qv' * v); n1 = norm(v)
            v .-= Qv * (Qv' * v); nv = norm(v)
            (nv < 1e-6 || nv < 0.5 * n1) && continue
            v ./= nv
            r += 1; Q[:, r] .= v; added += 1
            for c in 1:np1
                w = H.pieces[c](v); c == 1 && (w .-= σ .* v)
                W[c][:, r] .= w
            end
        end
        added == 0 && break
        push!(pts, copy(θ))
        Qr = view(Q, :, 1:r); Wr = [view(W[c], :, 1:r) for c in 1:np1]
        Ar = [Qr' * Wr[c] for c in 1:np1]
        Br = [Wr[a]' * Wr[b] for a in 1:np1, b in 1:np1]
        worst, θn, ybest = -1.0, θ, nothing
        for c in cands
            cc = (1.0, c...)
            F = eigen(Symmetric(sum(cc[j2] .* Ar[j2] for j2 in 1:np1)))
            Bm = sum(cc[a] * cc[b] .* Br[a, b] for a in 1:np1, b in 1:np1)
            for j2 in 1:min(k, length(F.values))
                y = F.vectors[:, j2]
                r2 = max(dot(y, Bm * y) - F.values[j2]^2, 0.0)
                r2 > worst && (worst = r2; θn = c; ybest = F.vectors[:, 1])
            end
        end
                res_prev = res; res = sqrt(worst)
        verbose && @printf("    r=%2d  certified residual %.2e\n", r, res)
        res < tol && break

        # Optional second stopping rule, on an observable rather than on the
        # residual: stop once the largest relative change in `monitor` between
        # consecutive snapshots falls below `monitor_tol`.  `monitor` is handed
        # the emulator built from the snapshots so far and returns the values
        # it wants watched.
        if monitor !== nothing
            partial = Emulator([Matrix(a) for a in Ar], k, copy(pts), dim, res,
                               Matrix(Qr))
            monitor_now = monitor(partial)
                        if monitor_prev !== nothing
                change = maximum(abs.(monitor_now .- monitor_prev) ./
                                 max.(abs.(monitor_now), eps()))
                verbose && @printf("    r=%2d  relative change %.2e\n", r, change)
                monitor_prev = monitor_now
                if change < monitor_tol
                    # The change at rank r compares r with r-1, so falling below
                    # tolerance certifies rank r-1: the last snapshot only served
                    # to show that the basis before it had already converged.
                    # Drop it, as the figure scripts for the papers do.
                    r = r_before
                    pop!(pts)
                    res = res_prev
                    break
                end
            else
                monitor_prev = monitor_now
            end
        end
        θ = θn
        # warm start: the reduced prediction at the next point is already a
        # good approximation to the exact vector there.
        guess = ybest === nothing ? nothing : view(Q, :, 1:r) * ybest
    end
    Qf = view(Q, :, 1:r)
    orth = maximum(abs.(Qf' * Qf - I))
    orth > 1e-10 && error("snapshot basis lost orthogonality ($orth)")
    Wf = [reduce(hcat, [H.pieces[c](Q[:, i]) for i in 1:r]) for c in 1:np1]
    # Matrix(Qf) copies the r used columns out of the preallocated block, so the
    # unused remainder can be freed.  Observables need this basis; the energy
    # does not, being a Ritz value of the projected matrices alone.
    Emulator([Qf' * Wf[c] for c in 1:np1], k, pts, dim, res, Matrix(Qf))
end
