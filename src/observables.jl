# ---------------------------------------------------------------------------
#  Observables on the reduced basis.
#
#  The energy is a Ritz value and comes straight out of the reduced eigenproblem.
#  An observable does not: it needs the snapshot basis itself, because what is
#  wanted is a norm in the full space.  For a symmetry-odd operator O the
#  quantity that locates a transition is the second moment
#
#      m^2(θ) = || O |GS(θ)> ||^2 = y(θ)' G y(θ),     G = (O Ψ)' (O Ψ),
#
#  with Ψ the stored snapshot basis and y the reduced ground state.  G is
#  assembled once, from r applications of O, after which m^2 costs nothing
#  beyond the reduced eigensolve already performed.
# ---------------------------------------------------------------------------
using LinearAlgebra
using .JScheme: Sector, eigensystem, orbit, rme_create1, rme_annih1, phase, sixj,
                O2Sector, o2_eigensystem, ninej

"""
    apply_order_param(src, dst, x)

Apply the Ising order parameter

    O = sum_m ( c†_{+,m} c_{-,m} + c†_{-,m} c_{+,m} )

to the coefficient vector `x` of the `(L, z2)` sector `src`, returning the
result in the `(L, -z2)` sector `dst`.  `O` is odd under the flavour exchange,
so it maps one parity block onto the other.  In the sigma^x basis this operator
is `sum_m sigma^z_m`, the magnetization whose second moment is the order
parameter.

`src` and `dst` must be built with the same `N`, `L` and pseudopotentials.
"""
function apply_order_param(src::Sector, dst::Sector, x::AbstractVector)
    N = src.N; orb = orbit(N); tj = N - 1; two_L = 2 * src.L
    v = zeros(dst.dim)

    # index the destination channels by particle number, so the n_- ± 1 lookup
    # below is a dictionary hit rather than a scan over every channel
    by_n = Dict{Int,Vector{NTuple{5,Int}}}()
    for ch in dst.channels
        push!(get!(() -> NTuple{5,Int}[], by_n, ch[1]), ch)
    end

    for (n_minus, two_Jp, two_Jm, mult_p, mult_m) in src.channels
        n_plus = N - n_minus
        xoff = src.offset[(n_minus, two_Jp, two_Jm)]
        X = reshape(view(x, xoff+1:xoff+mult_p*mult_m), mult_m, mult_p)

        #  :pm = c†_+ c_-  (n_- -> n_- - 1) ;  :mp = c†_- c_+  (n_- -> n_- + 1)
        for (branch, dn, tensor_sign) in ((:pm, -1, phase(tj)), (:mp, +1, -1.0))
            # the n_- ± 1 lookup doubles as the n_- >= 1 / n_+ >= 1 guard
            for (n2, Jp2, Jm2, mp2, mm2) in get(by_n, n_minus + dn,
                                                NTuple{5,Int}[])
                if branch === :pm
                    rp = rme_create1(orb, n_plus,  Jp2, two_Jp)
                    rm = rme_annih1( orb, n_minus, Jm2, two_Jm)
                else
                    rp = rme_annih1( orb, n_plus,  Jp2, two_Jp)
                    rm = rme_create1(orb, n_minus, Jm2, two_Jm)
                end
                (rp === nothing || rm === nothing) && continue
                c = phase(n_plus) * tensor_sign *
                    phase((two_Jp + Jm2 + two_L + tj) ÷ 2) *
                    sixj(Jp2, Jm2, two_L, two_Jm, two_Jp, tj)
                c == 0.0 && continue
                voff = dst.offset[(n2, Jp2, Jm2)]
                V = reshape(view(v, voff+1:voff+mp2*mm2), mm2, mp2)
                V .+= c .* (rm * X * transpose(rp))
            end
        end
    end
    return v
end

"""
    ising_order_param_gram(em, N; L = 0, z2 = +1, ps_pot = [4.75, 1.0])

Assemble `G = (O Ψ)' (O Ψ)` for the Ising order parameter, from the snapshot
basis `Ψ` that `em` carries.  Costs `r` applications of `O`, once; afterwards
[`reduced_expectation`](@ref) gives `m^2` at any coupling for free.

The sector basis depends only on `(N, L, z2)`, so `ps_pot` merely has to be the
length-2 form the sector constructor expects.

# Example

```julia
julia> H, solve = jscheme_ising_affine(8, 0, +1);

julia> em = greedy(H, solve, [(3.8, 5.8), (3.02, 3.32)]; k = 1, tol = 1e-9);

julia> G = ising_order_param_gram(em, 8);

julia> reduced_expectation(em, G, (4.75, 3.16))          # m^2 there
```
"""
function ising_order_param_gram(em::Emulator, N; L = 0, z2 = +1,
                                ps_pot = [4.75, 1.0])
    src = Sector(N, 0.0, ps_pot, L,  z2)
    dst = Sector(N, 0.0, ps_pot, L, -z2)
    size(em.basis, 1) == src.dim || error(
        "emulator dimension $(size(em.basis, 1)) does not match the " *
        "(L=$L, z2=$z2) sector at N=$N, which holds $(src.dim) states")
    r = size(em.basis, 2)
    M = reduce(hcat, [apply_order_param(src, dst, view(em.basis, :, c))
                      for c in 1:r])
    return Matrix(Symmetric(M' * M))
end

"""
    reduced_expectation(em, G, θ; root = 1)

`y' G y` for the `root`-th reduced eigenvector at coupling `θ`, with `G` from
[`ising_order_param_gram`](@ref).  One reduced eigensolve, no full-space work.
"""
function reduced_expectation(em::Emulator, G::AbstractMatrix, θ; root::Int = 1)
    _, vectors = emulate_state(em, θ; k = root)
    y = view(vectors, :, root)
    return dot(y, G, y)
end

"""
    ising_order_parameter(N, h, ps_pot; L = 0, z2 = 1, root = 1)

The exact route, for checking the emulated value: diagonalize the `(L, z2)`
sector and return `(energy, m^2)` of its `root`-th level.
"""
function ising_order_parameter(N, h, ps_pot::Vector{Float64};
                               L = 0, z2 = 1, root = 1, dense_limit = 4000)
    s0 = Sector(N, h, ps_pot, L, z2)
    s1 = Sector(N, h, ps_pot, L, -z2)
    values, vectors = eigensystem(s0; k = root, dense_limit = dense_limit)
    return values[root],
           sum(abs2, apply_order_param(s0, s1, view(vectors, :, root)))
end

# ---------------------------------------------------------------------------
#  The same construction for the O(2) model.  There the order parameter raises
#  the O(2) charge, mapping (L, Q) onto (L, Q+1), and <m> vanishes on a finite
#  sphere by the symmetry that is about to break, so m^2 is again the quantity
#  that locates the transition.
# ---------------------------------------------------------------------------

# Rank-2j coupled scalar [A^k(pair) x B^k(0)]^{K=0} in the ((J+ J-) Jc, J0) L
# basis.  Written out rather than reusing chain_scalar_*, which is validated
# only at even rank and whose scalar leg is the (-1)^q dot-product form, with
# no meaning at the half-integer rank needed here.
_op_p0(J0o, Jpo, Jm, Jco, J0i, Jpi, Jci, two_L, two_k) =
    sqrt((Jci + 1.0) * (Jco + 1.0) * (two_k + 1.0) * (Jm + 1.0)) *
    ninej(Jpi, Jm, Jci, two_k, 0, two_k, Jpo, Jm, Jco) *
    sqrt(two_L + 1.0) *
    ninej(Jci, J0i, two_L, two_k, two_k, 0, Jco, J0o, two_L)

_op_m0(J0o, Jp, Jmo, Jco, J0i, Jmi, Jci, two_L, two_k) =
    sqrt((Jci + 1.0) * (Jco + 1.0) * (two_k + 1.0) * (Jp + 1.0)) *
    ninej(Jp, Jmi, Jci, 0, two_k, two_k, Jp, Jmo, Jco) *
    sqrt(two_L + 1.0) *
    ninej(Jci, J0i, two_L, two_k, two_k, 0, Jco, J0o, two_L)

"""
    apply_order_param(src::O2Sector, dst::O2Sector, x)

Apply the O(2) order parameter, which raises the charge by one, to the
coefficient vector `x` of the `(L, Q)` sector `src`, returning the result in
the `(L, Q+1)` sector `dst`.  Both sectors must share `N`, `L` and `ps_pot`.
"""
function apply_order_param(src::O2Sector, dst::O2Sector, x::AbstractVector)
    N = src.N; orb = orbit(N); tj = N - 1; two_L = 2 * src.L
    v = zeros(dst.dim)
    for (np, nm, J0, Jp, Jm, Jc, m0, mp, mm) in src.channels
        n0 = N - np - nm
        xoff = src.offset[(np, nm, J0, Jp, Jm, Jc)]
        X = reshape(view(x, xoff+1:xoff+m0*mp*mm), mm, mp, m0)

        # ---- c†_+ c_0 :  n_+ -> n_+ + 1, n_0 -> n_0 - 1, J_- a spectator
        if n0 >= 1
            str = phase(np + nm) * phase(tj)   # fermion string x [AxB] ordering
            for (np2, nm2, J0o, Jpo, Jmo, Jco, m0o, mpo, mmo) in dst.channels
                (np2 == np + 1 && nm2 == nm && Jmo == Jm) || continue
                rp = rme_create1(orb, np, Jpo, Jp); rp === nothing && continue
                r0 = rme_annih1(orb, n0, J0o, J0);  r0 === nothing && continue
                c = str * sqrt(tj + 1.0) *
                    _op_p0(J0o, Jpo, Jm, Jco, J0, Jp, Jc, two_L, tj)
                c == 0.0 && continue
                voff = dst.offset[(np2, nm2, J0o, Jpo, Jmo, Jco)]
                V = reshape(view(v, voff+1:voff+m0o*mpo*mmo), mmo, mpo, m0o)
                for a0 in 1:m0, a0o in 1:m0o
                    w = c * r0[a0o, a0]; w == 0.0 && continue
                    @views V[:, :, a0o] .+= w .* (X[:, :, a0] * transpose(rp))
                end
            end
        end

        # ---- c†_0 c_- :  n_- -> n_- - 1, n_0 -> n_0 + 1, J_+ a spectator
        if nm >= 1
            str = phase(nm + 1)
            for (np2, nm2, J0o, Jpo, Jmo, Jco, m0o, mpo, mmo) in dst.channels
                (np2 == np && nm2 == nm - 1 && Jpo == Jp) || continue
                rm = rme_annih1(orb, nm, Jmo, Jm);  rm === nothing && continue
                r0 = rme_create1(orb, n0, J0o, J0); r0 === nothing && continue
                c = str * sqrt(tj + 1.0) *
                    _op_m0(J0o, Jp, Jmo, Jco, J0, Jm, Jc, two_L, tj)
                c == 0.0 && continue
                voff = dst.offset[(np2, nm2, J0o, Jpo, Jmo, Jco)]
                V = reshape(view(v, voff+1:voff+m0o*mpo*mmo), mmo, mpo, m0o)
                for a0 in 1:m0, a0o in 1:m0o
                    w = c * r0[a0o, a0]; w == 0.0 && continue
                    @views V[:, :, a0o] .+= w .* (rm * X[:, :, a0])
                end
            end
        end
    end
    return v
end

"""
    o2_order_param_gram(em, N; L = 0, Q = 0, ps_pot = [4.0, 1.0])

`G = (O Ψ)' (O Ψ)` for the O(2) order parameter, from the snapshot basis that
`em` carries.  Pair with [`reduced_expectation`](@ref), exactly as in the Ising
case.
"""
function o2_order_param_gram(em::Emulator, N; L = 0, Q = 0,
                             ps_pot = [4.0, 1.0])
    src = O2Sector(N, 0.0, ps_pot, L, Q)
    dst = O2Sector(N, 0.0, ps_pot, L, Q + 1)
    size(em.basis, 1) == src.dim || error(
        "emulator dimension $(size(em.basis, 1)) does not match the " *
        "(L=$L, Q=$Q) sector at N=$N, which holds $(src.dim) states")
    M = reduce(hcat, [apply_order_param(src, dst, view(em.basis, :, c))
                      for c in 1:size(em.basis, 2)])
    return Matrix(Symmetric(M' * M))
end

"""
    o2_order_parameter(N, D, ps_pot; L = 0, Q = 0, root = 1)

The exact route for the O(2) model: diagonalize the `(L, Q)` sector and return
`(energy, m^2)` of its `root`-th level.
"""
function o2_order_parameter(N, D, ps_pot::Vector{Float64};
                            L = 0, Q = 0, root = 1, dense_limit = 3000)
    s0 = O2Sector(N, D, ps_pot, L, Q)
    s1 = O2Sector(N, D, ps_pot, L, Q + 1)
    values, vectors = o2_eigensystem(s0; k = root, dense_limit = dense_limit)
    return values[root],
           sum(abs2, apply_order_param(s0, s1, view(vectors, :, root)))
end
