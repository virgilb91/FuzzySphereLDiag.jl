"""
    JScheme

Exact diagonalisation of the fuzzy-sphere Ising model in the J-scheme --
a Julia port of `jscheme.py` from this package, written so it could be
contributed to FuzzifiED (https://docs.fuzzified.world) as an exact-L
alternative to its m-scheme ED at moderate and large N.

The basis is |[(n_+, J_+, a_+) x (n_-, J_-, a_-)] L>: good total angular
momentum L by construction and exact Z2 = (-1)^{n_-}.  Every two-body term
factorises NuShellX-style (Brown & Rae, NDS 120, 115 (2014)) into per-orbit
reduced matrix elements tied by a 6j, so the Lanczos matvec is a sum of small
dense products -- which in Julia is a plain threaded loop: the shape-batching
and embedded C kernel that the Python version needs to escape its interpreter
have no reason to exist here.

House rule carried over: no hand-derived phase is trusted.  The Pandya and
pair-hopping coefficients are projected numerically from the m-scheme tensors
with the residual asserted, and the scalar-product 6j formula ships with the
numeric verifier that fitted it (`verify_scalar_formula`), run in `validate()`.
The end-to-end check is 30 reference energies measured independently with an
independent Python implementation (exact ED at N <= 8) and with FuzzifiED --
both agree with this solver to every printed digit.

Dependencies: LinearAlgebra (stdlib) and Arpack for the
iterative eigensolver (dense fallback below `dense_limit`).

Usage, as a submodule of FuzzySphereLDiag:
  using FuzzySphereLDiag
  const J = FuzzySphereLDiag.JScheme
  J.solve(16, 3.153; L = 2, z2 = +1, k = 2)
  J.validate()
"""
module JScheme

using LinearAlgebra
using Printf

const HAVE_ARPACK = Base.find_package("Arpack") !== nothing
@static if Base.find_package("Arpack") !== nothing
    import Arpack
end

phase(n::Integer) = isodd(n) ? -1.0 : 1.0

# =============================================================================
# 1. Angular momentum: 3j, CG, exact-rational 6j (doubled arguments throughout)
# =============================================================================

const _LOGFACT = Float64[0.0]

function logfact(n::Integer)
    n < 0 && return Inf
    while length(_LOGFACT) <= n
        push!(_LOGFACT, _LOGFACT[end] + log(length(_LOGFACT)))
    end
    return _LOGFACT[n+1]
end

# pre-extend once at load: the table must not grow under threads (the
# threaded sector build calls sixj -> logfact concurrently)
logfact(2000)

function wigner3j(j1, j2, j3, m1, m2, m3)
    (m1 + m2 + m3 != 0 || abs(m1) > j1 || abs(m2) > j2 || abs(m3) > j3) &&
        return 0.0
    (isodd(j1 + j2 + j3) || j3 > j1 + j2 || j3 < abs(j1 - j2)) && return 0.0
    parts = (j1 + m1, j1 - m1, j2 + m2, j2 - m2, j3 + m3, j3 - m3)
    any(isodd, parts) && return 0.0
    a, b, c = (j1 + j2 - j3) ÷ 2, (j1 - j2 + j3) ÷ 2, (-j1 + j2 + j3) ÷ 2
    min(a, b, c) < 0 && return 0.0
    triangle = logfact(a) + logfact(b) + logfact(c) -
               logfact((j1 + j2 + j3) ÷ 2 + 1)
    numerator = sum(logfact(p ÷ 2) for p in parts)
    isodd(j1 - j2 - m3) && return 0.0
    prefactor = phase((j1 - j2 - m3) ÷ 2) * exp(0.5 * (triangle + numerator))
    zb, zc = (j1 - m1) ÷ 2, (j2 + m2) ÷ 2
    zd, ze = (j3 - j2 + m1) ÷ 2, (j3 - j1 - m2) ÷ 2
    total = 0.0
    for z in max(0, -zd, -ze):min(a, zb, zc)
        total += phase(z) * exp(-(logfact(z) + logfact(a - z) +
                                  logfact(zb - z) + logfact(zc - z) +
                                  logfact(zd + z) + logfact(ze + z)))
    end
    return prefactor * total
end

const _CG_CACHE = Dict{NTuple{6,Int},Float64}()

function cg(j1, m1, j2, m2, J, M)
    m1 + m2 != M && return 0.0
    key = (j1, m1, j2, m2, J, M)
    get!(_CG_CACHE, key) do
        phase((j1 - j2 + M) ÷ 2) * sqrt(J + 1.0) *
            wigner3j(j1, j2, J, m1, m2, -M)
    end
end

# 6j caching under threads: a shared cache that is READ-ONLY while threads
# run, plus one overflow cache per thread for keys not yet shared.  After a
# threaded build, merge_sixj_caches!() folds the overflows into the shared
# cache (single-threaded), so later sectors hit it without duplicate work.
const _SIXJ_SHARED = Dict{NTuple{6,Int},Float64}()
const _SIXJ_CACHES = [Dict{NTuple{6,Int},Float64}()
                      for _ in 1:Threads.maxthreadid()+8]

function merge_sixj_caches!()
    for c in _SIXJ_CACHES
        merge!(_SIXJ_SHARED, c)
        empty!(c)
    end
end

"""Canonical representative under the 24 classical 6j symmetries (column
permutations and pair flips in two columns) -- shrinks the cache key space
~15x.  The symmetry claim is verified numerically against `_sixj_compute`
in `o2_validate` (house rule: no hand-taken identity untested)."""
function _sixj_canonical(j1, j2, j3, j4, j5, j6)
    best = (j1, j2, j3, j4, j5, j6)
    cols = ((j1, j4), (j2, j5), (j3, j6))
    for p in ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2),
              (3, 2, 1))
        a, b, c = cols[p[1]], cols[p[2]], cols[p[3]]
        for flip in 0:3          # flip the pair in all but one column
            aa = flip == 2 || flip == 3 ? (a[2], a[1]) : a
            bb = flip == 1 || flip == 3 ? (b[2], b[1]) : b
            cc = flip == 1 || flip == 2 ? (c[2], c[1]) : c
            key = (aa[1], bb[1], cc[1], aa[2], bb[2], cc[2])
            key < best && (best = key)
        end
    end
    return best
end

"""Racah 6j in exact rational arithmetic; the alternating sum cancels
catastrophically in floating point at the j ~ 15 this file cares about."""
function sixj(a1, a2, a3, a4, a5, a6)
    key = _sixj_canonical(a1, a2, a3, a4, a5, a6)
    shared = get(_SIXJ_SHARED, key, NaN)
    isnan(shared) || return shared
    _SIXJ_CACHE = _SIXJ_CACHES[Threads.threadid()]
    haskey(_SIXJ_CACHE, key) && return _SIXJ_CACHE[key]
    return _SIXJ_CACHE[key] = _sixj_compute(key...)
end

function _sixj_compute(j1, j2, j3, j4, j5, j6)
    triangles = ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3))
    for (a, b, c) in triangles
        if isodd(a + b + c) || c > a + b || c < abs(a - b)
            return 0.0
        end
    end
    log_prefactor = 0.0
    for (a, b, c) in triangles
        log_prefactor += 0.5 * (logfact((a + b - c) ÷ 2) +
                                logfact((a - b + c) ÷ 2) +
                                logfact((-a + b + c) ÷ 2) -
                                logfact((a + b + c) ÷ 2 + 1))
    end
    lows = ((j1 + j2 + j3) ÷ 2, (j1 + j5 + j6) ÷ 2,
            (j4 + j2 + j6) ÷ 2, (j4 + j5 + j3) ÷ 2)
    highs = ((j1 + j2 + j4 + j5) ÷ 2, (j2 + j3 + j5 + j6) ÷ 2,
             (j3 + j1 + j6 + j4) ÷ 2)
    total = Rational{BigInt}(0)
    for t in maximum(lows):minimum(highs)
        denominator = BigInt(1)
        for low in lows
            denominator *= factorial(big(t - low))
        end
        for high in highs
            denominator *= factorial(big(high - t))
        end
        total += Rational{BigInt}((-1)^t * factorial(big(t + 1)), denominator)
    end
    total == 0 && return 0.0
    sign = total > 0 ? 1.0 : -1.0
    log_total = log(abs(numerator(total))) - log(denominator(total))
    return sign * exp(log_prefactor + Float64(log_total))
end

# =============================================================================
# 2. Single-orbit tables
# =============================================================================

"""Apply a fermion string (reading order; last element acts first).
Operators are (dagger::Bool, orbital::Int 0-based).  Returns (state, sign)."""
function apply_ops(state::Int, ops)
    sign = 1
    for i in length(ops):-1:1
        dagger, orbital = ops[i]
        bit = 1 << orbital
        occupied = state & bit != 0
        (dagger == occupied) && return 0, 0
        sign *= isodd(count_ones(state & (bit - 1))) ? -1 : 1
        state = dagger ? state | bit : state & ~bit
    end
    return state, sign
end

mutable struct Orbit
    N::Int
    two_j::Int
    blocks::Dict{Int,Dict{Int,Vector{Int}}}
    hw::Dict{Int,Dict{Int,Matrix{Float64}}}
    rme_t::Dict{NTuple{4,Int},Union{Matrix{Float64},Nothing}}
    rme_pair::Dict{NTuple{4,Int},Union{Matrix{Float64},Nothing}}
    rme_y::Dict{NTuple{4,Int},Union{Matrix{Float64},Nothing}}
    w::Dict{NTuple{3,Int},Matrix{Float64}}
end

Orbit(N) = Orbit(N, N - 1, Dict(), Dict(), Dict(), Dict(), Dict(), Dict())

const _ORBITS = Dict{Int,Orbit}()
orbit(N) = get!(() -> Orbit(N), _ORBITS, N)

"""{two_M => sorted bitmasks} for n particles, by Gosper enumeration."""
function blocks(orb::Orbit, n::Int)
    get!(orb.blocks, n) do
        out = Dict{Int,Vector{Int}}()
        if n == 0
            out[0] = [0]
            return out
        end
        mask = (1 << n) - 1
        limit = 1 << orb.N
        while mask < limit
            two_M = 0
            bits = mask
            while bits != 0
                k = trailing_zeros(bits)
                two_M += 2k - orb.two_j
                bits &= bits - 1
            end
            push!(get!(() -> Int[], out, two_M), mask)
            # Gosper's hack: next integer with the same popcount
            low = mask & -mask
            ripple = mask + low
            mask = ripple | (((mask ⊻ ripple) >> 2) ÷ low)
        end
        foreach(sort!, values(out))
        out
    end
end

block_dim(orb, n, two_M) = length(get(blocks(orb, n), two_M, Int[]))

function block_operator(orb::Orbit, terms, n_in, two_M_in, n_out, two_M_out)
    source = get(blocks(orb, n_in), two_M_in, Int[])
    target = get(blocks(orb, n_out), two_M_out, Int[])
    index = Dict(s => i for (i, s) in enumerate(target))
    matrix = zeros(length(target), length(source))
    for (column, state) in enumerate(source)
        for (coefficient, ops) in terms
            out, sign = apply_ops(state, ops)
            if sign != 0
                row = get(index, out, 0)
                row != 0 && (matrix[row, column] += sign * coefficient)
            end
        end
    end
    return matrix
end

"""nullspace via full SVD, falling back from the divide-and-conquer LAPACK
driver (dgesdd, which occasionally fails to converge on blocks this large --
first seen on an N = 20 M-block) to QR iteration (dgesvd, slower, robust)."""
function robust_nullspace(A::Matrix{Float64})
    F = try
        svd(A; full = true)
    catch err
        err isa LinearAlgebra.LAPACKException || rethrow()
        svd(A; full = true, alg = LinearAlgebra.QRIteration())
    end
    tol = maximum(size(A)) * eps(Float64) *
          (isempty(F.S) ? 1.0 : first(F.S))
    rank = count(>(tol), F.S)
    return F.V[:, rank+1:end]
end

"""Highest-weight vectors: {two_J => (d_block x mult)}, kernel of J+."""
function hw(orb::Orbit, n::Int)
    get!(orb.hw, n) do
        out = Dict{Int,Matrix{Float64}}()
        jj = 0.25 * orb.two_j * (orb.two_j + 2)
        for (two_M, states) in sort(collect(blocks(orb, n)); by = first)
            two_M < 0 && continue
            d_up = block_dim(orb, n, two_M + 2)
            length(states) == d_up && continue
            if d_up == 0
                out[two_M] = Matrix{Float64}(I, length(states), length(states))
            else
                terms = [(sqrt(jj - 0.25 * (2k - orb.two_j) *
                                    (2k - orb.two_j + 2)),
                          [(true, k + 1), (false, k)])
                         for k in 0:orb.N-2]
                raising = block_operator(orb, terms, n, two_M, n, two_M + 2)
                out[two_M] = robust_nullspace(raising)
            end
        end
        out
    end
end

multiplets(orb::Orbit, n) = Dict(J => size(m, 2) for (J, m) in hw(orb, n))

# =============================================================================
# 3. Tensor operators and reduced matrix elements
#
# Wigner-Eckart convention:  <J'M'|T^k_q|JM> = C(JM kq|J'M') <J'||T||J>
#                                              / sqrt(2J'+1),
# extracted at the stretched component (M = J, M' = J'), which never vanishes.
# =============================================================================

function t_terms(N, two_lam, two_mu)
    two_j = N - 1
    out = Tuple{Float64,Vector{Tuple{Bool,Int}}}[]
    for k1 in 0:N-1
        two_m1 = 2k1 - two_j
        two_m2 = two_m1 - two_mu
        k2 = (two_m2 + two_j) ÷ 2
        (0 <= k2 < N && iseven(two_m2 + two_j)) || continue
        value = cg(two_j, two_m1, two_j, -two_m2, two_lam, two_mu) *
                phase((two_j + two_m2) ÷ 2)
        value != 0 && push!(out, (value, [(true, k1), (false, k2)]))
    end
    return out
end

function pair_create_terms(N, two_J0, two_mu)
    two_j = N - 1
    out = Tuple{Float64,Vector{Tuple{Bool,Int}}}[]
    for k1 in 0:N-1
        two_m1 = 2k1 - two_j
        two_m2 = two_mu - two_m1
        k2 = (two_m2 + two_j) ÷ 2
        (0 <= k2 < N && k1 != k2 && iseven(two_m2 + two_j)) || continue
        value = sqrt(0.5) * cg(two_j, two_m1, two_j, two_m2, two_J0, two_mu)
        value != 0 && push!(out, (value, [(true, k1), (true, k2)]))
    end
    return out
end

function pair_y_terms(N, two_J0, two_nu)
    two_j = N - 1
    sign = phase((two_J0 - two_nu) ÷ 2)
    out = Tuple{Float64,Vector{Tuple{Bool,Int}}}[]
    for k3 in 0:N-1
        two_m3 = 2k3 - two_j
        two_m4 = -two_nu - two_m3
        k4 = (two_m4 + two_j) ÷ 2
        (0 <= k4 < N && k3 != k4 && iseven(two_m4 + two_j)) || continue
        value = sign * sqrt(0.5) *
                cg(two_j, two_m3, two_j, two_m4, two_J0, -two_nu)
        value != 0 && push!(out, (value, [(false, k4), (false, k3)]))
    end
    return out
end

function stretched(orb::Orbit, terms, n_in, two_J, n_out, two_Jp, two_k)
    abs(two_Jp - two_J) <= two_k <= two_Jp + two_J || return nothing
    hw_in = get(hw(orb, n_in), two_J, nothing)
    hw_out = get(hw(orb, n_out), two_Jp, nothing)
    (hw_in === nothing || hw_out === nothing) && return nothing
    op = block_operator(orb, terms, n_in, two_J, n_out, two_Jp)
    matrix = hw_out' * (op * hw_in)
    c = cg(two_J, two_J, two_k, two_Jp - two_J, two_Jp, two_Jp)
    return matrix .* (sqrt(two_Jp + 1.0) / c)
end

rme_t(orb, n, two_lam, two_Jp, two_J) =
    get!(() -> stretched(orb, t_terms(orb.N, two_lam, two_Jp - two_J),
                         n, two_J, n, two_Jp, two_lam),
         orb.rme_t, (n, two_lam, two_Jp, two_J))

rme_pair(orb, n_top, two_J0, two_Jp, two_J) =
    get!(() -> stretched(orb, pair_create_terms(orb.N, two_J0,
                                                two_Jp - two_J),
                         n_top - 2, two_J, n_top, two_Jp, two_J0),
         orb.rme_pair, (n_top, two_J0, two_Jp, two_J))

rme_y(orb, n_in, two_J0, two_Jp, two_J) =
    get!(() -> stretched(orb, pair_y_terms(orb.N, two_J0, two_Jp - two_J),
                         n_in, two_J, n_in - 2, two_Jp, two_J0),
         orb.rme_y, (n_in, two_J0, two_Jp, two_J))

"""<n J a'|sum_M A+_M A_M|n J a> from the pair RMEs (checked in validate)."""
function pair_scalar(orb::Orbit, n, two_J, two_J0)
    get!(orb.w, (n, two_J, two_J0)) do
        mult = get(multiplets(orb, n), two_J, 0)
        total = zeros(mult, mult)
        if n >= 2
            for (two_Jpp, _) in multiplets(orb, n - 2)
                abs(two_J - two_Jpp) > two_J0 && continue
                rme = rme_pair(orb, n, two_J0, two_J, two_Jpp)
                rme === nothing && continue
                total .+= (rme * rme') ./ (two_J + 1.0)
            end
        end
        total
    end
end

# =============================================================================
# 4. Recoupling coefficients, derived by the program
# =============================================================================

function t_coefficient_matrix(N, two_lam, two_mu)
    matrix = zeros(N, N)
    for (value, ops) in t_terms(N, two_lam, two_mu)
        matrix[ops[1][2]+1, ops[2][2]+1] += value
    end
    return matrix
end

function pair_coefficient_matrix(terms, N)
    matrix = zeros(N, N)
    for (value, ops) in terms
        ka, kb = ops[1][2], ops[2][2]
        if ka < kb
            matrix[ka+1, kb+1] += value
        else
            matrix[kb+1, ka+1] -= value
        end
    end
    return matrix
end

const _PANDYA = Dict{Any,Dict{Int,Float64}}()

"""c_lam in  sum over (+-) channels of  weight_J sum_M A+_{(+-)JM} A_{(+-)JM}
             = sum_lam c_lam sum_mu (-1)^mu T^lam_mu(+) T^lam_{-mu}(-).
`channels` is a tuple of (two_J, weight) pairs -- one per even-l
pseudopotential with weight 2 V_l.  Projected from the m-scheme coefficient
tensors; residual asserted, so a wrong channel structure cannot survive."""
function pandya_coefficients(N::Int, channels::Tuple)
    get!(_PANDYA, (N, channels)) do
        two_j = N - 1
        G = zeros(N, N, N, N)                    # [k1, k3, k2, k4]
        for (two_J1, weight) in channels
            for k1 in 0:N-1, k2 in 0:N-1
                two_M = (2k1 - two_j) + (2k2 - two_j)
                abs(two_M) > two_J1 && continue
                left = cg(two_j, 2k1 - two_j, two_j, 2k2 - two_j,
                          two_J1, two_M)
                left == 0 && continue
                for k3 in 0:N-1
                    two_m4 = two_M - (2k3 - two_j)
                    k4 = (two_m4 + two_j) ÷ 2
                    (0 <= k4 < N && iseven(two_m4 + two_j)) || continue
                    right = cg(two_j, 2k3 - two_j, two_j, two_m4,
                               two_J1, two_M)
                    G[k1+1, k3+1, k2+1, k4+1] += weight * left * right
                end
            end
        end
        out = Dict{Int,Float64}()
        reconstruction = zeros(size(G))
        for two_lam in 0:2:2*two_j
            basis = zeros(size(G))
            for two_mu in -two_lam:2:two_lam
                plus = t_coefficient_matrix(N, two_lam, two_mu)
                minus = t_coefficient_matrix(N, two_lam, -two_mu)
                p = phase(two_mu ÷ 2)
                @inbounds for i4 in 1:N, i2 in 1:N, i3 in 1:N, i1 in 1:N
                    basis[i1, i3, i2, i4] += p * plus[i1, i3] * minus[i2, i4]
                end
            end
            norm2 = sum(abs2, basis)
            norm2 < 1e-14 && continue
            c = dot(vec(basis), vec(G)) / norm2
            if abs(c) > 1e-14
                out[two_lam] = c
                reconstruction .+= c .* basis
            end
        end
        residual = maximum(abs, G .- reconstruction)
        @assert residual < 1e-10 "Pandya projection failed: $residual"
        out
    end
end

const _PAIRHOP = Dict{Tuple{Int,Int},Float64}()

"""d in  sum_M A+_{(++)J0M} A_{(--)J0M}
         = d sum_mu (-1)^mu A+^{J0}_mu(+) Y^{J0}_{-mu}(-)."""
function pair_hop_coefficient(N::Int, two_J0::Int)
    get!(_PAIRHOP, (N, two_J0)) do
        two_j = N - 1
        G = zeros(N, N, N, N)
        for two_M in -two_J0:2:two_J0
            create = pair_coefficient_matrix(
                pair_create_terms(N, two_J0, two_M), N)
            annihilate_terms = Tuple{Float64,Vector{Tuple{Bool,Int}}}[]
            for k3 in 0:N-1
                two_m4 = two_M - (2k3 - two_j)
                k4 = (two_m4 + two_j) ÷ 2
                (0 <= k4 < N && k3 != k4 && iseven(two_m4 + two_j)) || continue
                value = sqrt(0.5) *
                        cg(two_j, 2k3 - two_j, two_j, two_m4, two_J0, two_M)
                value != 0 &&
                    push!(annihilate_terms, (value, [(false, k4), (false, k3)]))
            end
            annihilate = pair_coefficient_matrix(annihilate_terms, N)
            @inbounds for i4 in 1:N, i3 in 1:N, i2 in 1:N, i1 in 1:N
                G[i1, i2, i3, i4] += create[i1, i2] * annihilate[i3, i4]
            end
        end
        basis = zeros(size(G))
        for two_mu in -two_J0:2:two_J0
            create = pair_coefficient_matrix(
                pair_create_terms(N, two_J0, two_mu), N)
            y = pair_coefficient_matrix(pair_y_terms(N, two_J0, -two_mu), N)
            p = phase(two_mu ÷ 2)
            @inbounds for i4 in 1:N, i3 in 1:N, i2 in 1:N, i1 in 1:N
                basis[i1, i2, i3, i4] += p * create[i1, i2] * y[i3, i4]
            end
        end
        d = dot(vec(basis), vec(G)) / sum(abs2, basis)
        residual = maximum(abs, G .- d .* basis)
        @assert residual < 1e-10 "pair-hop projection failed: $residual"
        d
    end
end

"""coef in <[(j1'j2')L]|sum_q (-1)^q T^k_q(1) U^k_{-q}(2)|[(j1j2)L]>
= coef <j1'||T||j1><j2'||U||j2>.  The phase convention was FITTED against
explicit constructions in the python session; `verify_scalar_formula`
re-derives it numerically and is run by `validate()`."""
scalar_coef(two_j1p, two_j2p, two_j1, two_j2, two_L, two_k) =
    phase((two_j1 + two_j2p + two_L) ÷ 2) *
    sixj(two_j1p, two_j2p, two_L, two_j2, two_j1, two_k)

function numeric_scalar_coef(two_j1p, two_j2p, two_j1, two_j2, two_L, two_k)
    coupled(ja, jb, L, M) = Dict((ma, M - ma) => cg(ja, ma, jb, M - ma, L, M)
                                 for ma in -ja:2:ja
                                 if abs(M - ma) <= jb &&
                                    cg(ja, ma, jb, M - ma, L, M) != 0)
    bra = coupled(two_j1p, two_j2p, two_L, two_L)
    ket = coupled(two_j1, two_j2, two_L, two_L)
    total = 0.0
    for two_q in -two_k:2:two_k
        for ((m1, m2), amp_ket) in ket
            amp_bra = get(bra, (m1 + two_q, m2 - two_q), 0.0)
            amp_bra == 0.0 && continue
            t = cg(two_j1, m1, two_k, two_q, two_j1p, m1 + two_q)
            u = cg(two_j2, m2, two_k, -two_q, two_j2p, m2 - two_q)
            total += phase(two_q ÷ 2) * amp_bra * amp_ket * t * u
        end
    end
    return total / sqrt((two_j1p + 1.0) * (two_j2p + 1.0))
end

function verify_scalar_formula(; samples = 24, seed = 7)
    rng = Base.copy(Random.MersenneTwister(seed))
    worst = 0.0
    tried = 0
    while tried < samples
        two_k = 2 * rand(rng, 0:3)
        two_j1, two_j2 = rand(rng, 0:7), rand(rng, 0:7)
        isodd(two_j1 + two_j2) && continue
        two_j1p = two_j1 + 2 * rand(rng, -2:2)
        two_j2p = two_j2 + 2 * rand(rng, -2:2)
        (two_j1p < 0 || two_j2p < 0) && continue
        (abs(two_j1 - two_j1p) <= two_k <= two_j1 + two_j1p) || continue
        (abs(two_j2 - two_j2p) <= two_k <= two_j2 + two_j2p) || continue
        lo = max(abs(two_j1 - two_j2), abs(two_j1p - two_j2p))
        hi = min(two_j1 + two_j2, two_j1p + two_j2p)
        lo > hi && continue
        two_L = lo + 2 * rand(rng, 0:(hi-lo)÷2)
        tried += 1
        worst = max(worst,
                    abs(scalar_coef(two_j1p, two_j2p, two_j1, two_j2,
                                    two_L, two_k) -
                        numeric_scalar_coef(two_j1p, two_j2p, two_j1, two_j2,
                                            two_L, two_k)))
    end
    return worst
end

import Random

"""9j symbol as the standard 6j sum (all arguments doubled)."""
function ninej(a, b, c, d, e, f, g, h, i)
    lo = max(abs(a - i), abs(b - f), abs(d - h))
    hi = min(a + i, b + f, d + h)
    total = 0.0
    for x in lo:2:hi
        total += (x + 1) * phase(x) *
                 sixj(a, b, c, f, i, x) *
                 sixj(d, e, f, b, x, h) *
                 sixj(g, h, i, x, a, d)
    end
    return total
end

"""<n+1, J'||a^dag||n, J> and <n-1, J'||a~||n, J>: single-particle tensors of
rank j (odd fermion count -- used only by the cross-flavour observables,
never by the Hamiltonian)."""
rme_create1(orb, n, two_Jp, two_J) =
    get!(() -> stretched(orb, [(1.0, [(true,
        (two_Jp - two_J + orb.two_j) ÷ 2)])], n, two_J, n + 1, two_Jp,
        orb.two_j), orb.rme_pair, (-1000 - n, two_Jp, two_J, 1))

rme_annih1(orb, n, two_Jp, two_J) = get!(orb.rme_pair,
        (-2000 - n, two_Jp, two_J, 1)) do
    mu2 = two_Jp - two_J
    k = (-mu2 + orb.two_j) ÷ 2
    (0 <= k < orb.N) || return nothing
    stretched(orb, [(phase((orb.two_j - mu2) ÷ 2), [(false, k)])],
              n, two_J, n - 1, two_Jp, orb.two_j)
end

"""Apply the Z2-odd rank-lambda density tensor to an L = 0 state:
v = { [a^dag(+) (x) a~(-)]^lam + s_jw (-1)^p [a^dag(-) (x) a~(+)]^lam } |gs>,
returning the reduced vector in the (lam, -Z2) sector.  The coupled
coefficient uses the 9j form; the overall per-lambda constant and the
cross-flavour Jordan-Wigner sign rule are FITTED against the m-scheme oracle
(chi_oracle.py) -- validated, not derived."""
function apply_odd_tensor(src, dst, two_lam,
                          x::Vector{Float64}; jw_exponent = 1)
    orb = orbit(src.N)
    v = zeros(dst.dim)
    j2 = src.N - 1
    for (n_m, two_Jp, two_Jm, mp, mm) in src.channels
        xoff = src.offset[(n_m, two_Jp, two_Jm)]
        X = reshape(view(x, xoff+1:xoff+mp*mm), mm, mp)
        for (branch, dn) in ((:pm, -1), (:mp, +1))
            n_m_out = n_m + dn
            0 <= n_m_out <= src.N || continue
            for (n2, Jp2, Jm2, mp2, mm2) in dst.channels
                n2 == n_m_out || continue
                if branch == :pm
                    R1 = rme_create1(orb, src.N - n_m, Jp2, two_Jp)
                    R2 = rme_annih1(orb, n_m, Jm2, two_Jm)
                else
                    R1 = rme_annih1(orb, src.N - n_m, Jp2, two_Jp)
                    R2 = rme_create1(orb, n_m, Jm2, two_Jm)
                end
                (R1 === nothing || R2 === nothing) && continue
                nine = ninej(two_Jp, two_Jm, 0, j2, j2, two_lam,
                             Jp2, Jm2, two_lam)
                nine == 0.0 && continue
                # <(J'1 J'2)L'||[A x B]^K||(J1 J2)L> =
                #   sqrt((2L+1)(2L'+1)(2K+1)) * 9j * RME1 * RME2 ;
                # L = 0, L' = K = lam  =>  prefactor (2 lam + 1).
                coef = (two_lam + 1.0) * nine
                # candidate fermionic reordering phase, fitted vs the oracle
                if jw_exponent == 1
                    branch == :mp && (coef *= phase(n_m))
                elseif jw_exponent == 2
                    coef *= phase(branch == :pm ? n_m : src.N - n_m)
                elseif jw_exponent == 3
                    coef *= phase((two_Jp + two_Jm - Jp2 - Jm2) ÷ 2)
                elseif jw_exponent == 4
                    branch == :mp && (coef *= phase(two_lam ÷ 2))
                elseif jw_exponent == 5
                    branch == :mp && (coef *= -phase(two_lam ÷ 2))
                end
                voff = dst.offset[(n_m_out, Jp2, Jm2)]
                V = reshape(view(v, voff+1:voff+mp2*mm2), mm2, mp2)
                V .+= coef .* (R2 * X * R1')
            end
        end
    end
    return v
end

"""Static susceptibility (reduced units): chi = 2 <v|(H - E0)^{-1}|v> by
conjugate gradient on the positive-definite shifted sector Hamiltonian."""
function reduced_chi(dst, v::Vector{Float64}, E0; tol = 1e-8)
    buffers, tmps, ranges = workspace(dst)
    Hx = x -> matvec!(zeros(dst.dim), dst, x, buffers, tmps, ranges) .- E0 .* x
    x = zeros(dst.dim)
    r = copy(v); p = copy(r)
    rs = dot(r, r)
    for _ in 1:2000
        Ap = Hx(p)
        alpha = rs / dot(p, Ap)
        x .+= alpha .* p
        r .-= alpha .* Ap
        rs_new = dot(r, r)
        sqrt(rs_new) < tol * (1 + sqrt(dot(v, v))) && break
        p .= r .+ (rs_new / rs) .* p
        rs = rs_new
    end
    return 2 * dot(v, x), dot(v, v)
end

# =============================================================================
# 5. Sector: basis, factorised tasks, threaded matvec
#
# Block layout (column-major): within a channel the coefficients form the
# (mult_-, mult_+) matrix X with X[a_-, a_+], flattened columnwise, so
# vec(Y) = kron(R_+, R_-) vec(X)  and the task update is
# Y += coef * R_- * X * R_+'.
# =============================================================================

struct Task
    out_off::Int
    in_off::Int
    coef::Float64
    rp::Matrix{Float64}                          # (mult'_+, mult_+)
    rm::Matrix{Float64}                          # (mult'_-, mult_-)
end

struct Sector
    N::Int
    h::Float64
    L::Int
    z2::Int
    dim::Int
    channels::Vector{NTuple{5,Int}}              # (n_-, 2J_+, 2J_-, m_+, m_-)
    offset::Dict{NTuple{3,Int},Int}
    diagonal::Vector{Tuple{Int,Float64,Matrix{Float64},Matrix{Float64}}}
    tasks::Vector{Task}
end

Sector(N, h, V0::Real, V1::Real, L, z2; kwargs...) =
    Sector(N, h, [Float64(V0), Float64(V1)], L, z2; kwargs...)

"""General pseudopotential list: ps_pot[l+1] = V_l, FuzzifiED convention
(their `GetDenIntTerms` argument is `2 .* ps_pot`).  In the sigma^x basis the
even-l potentials feed the (+-) channels at odd pair-J = N-1-l and the odd-l
potentials feed the same-flavour channels at even pair-J -- the structure that
`validate` pins against FuzzifiED one l at a time."""
function Sector(N, h, ps_pot::Vector{Float64}, L, z2; n_minus_max = nothing)
    orb = orbit(N)
    two_L = 2L
    channels = NTuple{5,Int}[]
    offset = Dict{NTuple{3,Int},Int}()
    position = 0
    start = z2 > 0 ? 0 : 1
    stop = n_minus_max === nothing ? N : min(N, n_minus_max)
    for n_minus in start:2:stop
        plus = multiplets(orb, N - n_minus)
        minus = multiplets(orb, n_minus)
        for (two_Jp, mult_p) in sort(collect(plus); by = first)
            for (two_Jm, mult_m) in sort(collect(minus); by = first)
                abs(two_Jp - two_Jm) <= two_L <= two_Jp + two_Jm || continue
                push!(channels, (n_minus, two_Jp, two_Jm, mult_p, mult_m))
                offset[(n_minus, two_Jp, two_Jm)] = position
                position += mult_p * mult_m
            end
        end
    end

    # even-l pseudopotentials -> (+-) channels at odd pair-J, weight 2 V_l
    pm_channels = Tuple((2 * (N - 1 - l), 2.0 * ps_pot[l+1])
                        for l in 0:length(ps_pot)-1
                        if iseven(l) && ps_pot[l+1] != 0.0)
    c_lambda = isempty(pm_channels) ? Dict{Int,Float64}() :
               pandya_coefficients(N, pm_channels)
    # odd-l pseudopotentials -> same-flavour channels at even pair-J
    ff_channels = Tuple((2 * (N - 1 - l), ps_pot[l+1])
                        for l in 0:length(ps_pot)-1
                        if isodd(l) && ps_pot[l+1] != 0.0)
    diagonal = Tuple{Int,Float64,Matrix{Float64},Matrix{Float64}}[]
    tasks = Task[]
    by_n = Dict{Int,Vector{NTuple{5,Int}}}()
    for channel in channels
        push!(get!(() -> NTuple{5,Int}[], by_n, channel[1]), channel)
    end

    for (n_minus, two_Jp, two_Jm, mult_p, mult_m) in channels
        n_plus = N - n_minus
        scalar = -h * (n_plus - n_minus)
        w_plus = zeros(mult_p, mult_p)
        w_minus = zeros(mult_m, mult_m)
        for (two_J0, weight) in ff_channels
            w_plus .+= weight .* pair_scalar(orb, n_plus, two_Jp, two_J0)
            w_minus .+= weight .* pair_scalar(orb, n_minus, two_Jm, two_J0)
        end
        push!(diagonal, (offset[(n_minus, two_Jp, two_Jm)], scalar,
                         w_plus, w_minus))
    end

    # (+-)(+-): diagonal in n_-, scalar products of density tensors
    for (n_minus, group) in by_n
        n_plus = N - n_minus
        for out_ch in group, in_ch in group
            _, Jp_o, Jm_o, _, _ = out_ch
            _, Jp_i, Jm_i, _, _ = in_ch
            for (two_lam, c) in c_lambda
                (abs(Jp_o - Jp_i) <= two_lam &&
                 abs(Jm_o - Jm_i) <= two_lam) || continue
                r_plus = rme_t(orb, n_plus, two_lam, Jp_o, Jp_i)
                r_minus = rme_t(orb, n_minus, two_lam, Jm_o, Jm_i)
                (r_plus === nothing || r_minus === nothing) && continue
                coefficient = c * scalar_coef(Jp_o, Jm_o, Jp_i, Jm_i,
                                              two_L, two_lam)
                abs(coefficient) * maximum(abs, r_plus) *
                    maximum(abs, r_minus) < 1e-14 && continue
                push!(tasks, Task(offset[out_ch[1:3]], offset[in_ch[1:3]],
                                  coefficient, r_plus, r_minus))
            end
        end
    end

    # same-flavour pair hop, per even-J channel, both directions
    for (two_J0, weight) in ff_channels
        d_hop = pair_hop_coefficient(N, two_J0)
        for direction in (-2, 2)
            for in_ch in channels
                n_minus_i, Jp_i, Jm_i, _, _ = in_ch
                n_minus_o = n_minus_i + direction
                0 <= n_minus_o <= N || continue
                n_plus_i = N - n_minus_i
                for out_ch in get(by_n, n_minus_o, NTuple{5,Int}[])
                    _, Jp_o, Jm_o, _, _ = out_ch
                    (abs(Jp_o - Jp_i) <= two_J0 &&
                     abs(Jm_o - Jm_i) <= two_J0) || continue
                    if direction == -2
                        r_plus = rme_pair(orb, n_plus_i + 2, two_J0,
                                          Jp_o, Jp_i)
                        r_minus = rme_y(orb, n_minus_i, two_J0, Jm_o, Jm_i)
                    else
                        r_plus = rme_y(orb, n_plus_i, two_J0, Jp_o, Jp_i)
                        r_minus = rme_pair(orb, n_minus_i + 2, two_J0,
                                           Jm_o, Jm_i)
                    end
                    (r_plus === nothing || r_minus === nothing) && continue
                    coefficient = -weight * d_hop *
                                  scalar_coef(Jp_o, Jm_o, Jp_i, Jm_i,
                                              two_L, two_J0)
                    abs(coefficient) * maximum(abs, r_plus) *
                        maximum(abs, r_minus) < 1e-14 && continue
                    push!(tasks, Task(offset[out_ch[1:3]],
                                      offset[in_ch[1:3]],
                                      coefficient, r_plus, r_minus))
                end
            end
        end
    end
    return Sector(N, h, L, z2, position, channels, offset, diagonal, tasks)
end

# Hand-rolled kernels for the tiny per-task GEMMs: at typical block sizes
# (5-100) a BLAS call costs more in dispatch than in arithmetic, and calling
# OpenBLAS from many Julia threads contends on its internal pool -- measured
# at 137 ms/matvec via mul! against 34 ms for the equivalent C kernel.

"""T(c,b) = Rm(c,d) * X, X read column-major from x at in_off."""
@inline function _left_mul!(T, rm, x, in_off, b, d)
    c = size(rm, 1)
    @inbounds for col in 1:b
        for i in 1:c
            T[i, col] = 0.0
        end
        base = in_off + (col - 1) * d
        for l in 1:d
            v = x[base+l]
            v == 0.0 && continue
            @simd for i in 1:c
                T[i, col] += rm[i, l] * v
            end
        end
    end
end

"""buffer[out_off + .] += coef * T(c,b) * Rp'(b,a), column-major."""
@inline function _right_mul!(buffer, out_off, T, rp, coef, a, b, c)
    @inbounds for j in 1:a
        base = out_off + (j - 1) * c
        for l in 1:b
            v = coef * rp[j, l]
            v == 0.0 && continue
            @simd for i in 1:c
                buffer[base+i] += v * T[i, l]
            end
        end
    end
end

"""Threaded matvec: tasks partitioned over threads, each with a private
output buffer; the diagonal part is applied once at the end."""
function matvec!(y::AbstractVector{Float64}, sector::Sector,
                 x::AbstractVector{Float64}, buffers, tmps, ranges)
    nthreads = length(buffers)
    tasks = sector.tasks
    Threads.@threads :static for w in 1:nthreads
        buffer = buffers[w]
        tmp = tmps[w]
        fill!(buffer, 0.0)
        for t in ranges[w]
            task = tasks[t]
            a, b = size(task.rp)                 # mult'_+, mult_+
            c, d = size(task.rm)                 # mult'_-, mult_-
            T = reshape(view(tmp, 1:c*b), c, b)
            _left_mul!(T, task.rm, x, task.in_off, b, d)
            _right_mul!(buffer, task.out_off, T, task.rp, task.coef, a, b, c)
        end
    end
    fill!(y, 0.0)
    for buffer in buffers
        y .+= buffer
    end
    for (off, scalar, w_plus, w_minus) in sector.diagonal
        mp, mm = size(w_plus, 1), size(w_minus, 1)
        X = reshape(view(x, off+1:off+mp*mm), mm, mp)
        Y = reshape(view(y, off+1:off+mp*mm), mm, mp)
        Y .+= scalar .* X
        mul!(Y, w_minus, X, 1.0, 1.0)
        mul!(Y, X, w_plus, 1.0, 1.0)             # W symmetric
    end
    return y
end

function workspace(sector::Sector)
    nthreads = max(1, Threads.nthreads())
    max_tmp = isempty(sector.tasks) ? 1 :
              maximum(size(t.rm, 1) * size(t.rp, 2) for t in sector.tasks)
    # Partition tasks into contiguous ranges of equal FLOPs, not equal count:
    # block sizes vary by orders of magnitude within one sector, and an
    # equal-count split serialises on whichever thread drew the fat blocks.
    flops = [size(t.rm, 1) * size(t.rp, 2) * (size(t.rm, 2) + size(t.rp, 1))
             for t in sector.tasks]
    target = sum(flops) / nthreads
    ranges = UnitRange{Int}[]
    start, load = 1, 0.0
    for (t, f) in enumerate(flops)
        load += f
        if load >= target && length(ranges) < nthreads - 1
            push!(ranges, start:t)
            start, load = t + 1, 0.0
        end
    end
    push!(ranges, start:length(sector.tasks))
    while length(ranges) < nthreads
        push!(ranges, 1:0)
    end
    return ([zeros(sector.dim) for _ in 1:nthreads],
            [zeros(max_tmp) for _ in 1:nthreads], ranges)
end

function dense(sector::Sector)
    matrix = zeros(sector.dim, sector.dim)
    for (off, scalar, w_plus, w_minus) in sector.diagonal
        mp, mm = size(w_plus, 1), size(w_minus, 1)
        size_ = mp * mm
        block = kron(w_plus, Matrix{Float64}(I, mm, mm)) .+
                kron(Matrix{Float64}(I, mp, mp), w_minus) .+
                scalar .* Matrix{Float64}(I, size_, size_)
        matrix[off+1:off+size_, off+1:off+size_] .+= block
    end
    for task in sector.tasks
        block = task.coef .* kron(task.rp, task.rm)
        matrix[task.out_off+1:task.out_off+size(block, 1),
               task.in_off+1:task.in_off+size(block, 2)] .+= block
    end
    return matrix
end

struct SectorOperator <: AbstractMatrix{Float64}
    sector::Sector
    buffers::Vector{Vector{Float64}}
    tmps::Vector{Vector{Float64}}
    ranges::Vector{UnitRange{Int}}
end

Base.size(op::SectorOperator) = (op.sector.dim, op.sector.dim)
Base.size(op::SectorOperator, i::Int) = op.sector.dim
LinearAlgebra.mul!(y::AbstractVector, op::SectorOperator, x::AbstractVector) =
    matvec!(y, op.sector, x, op.buffers, op.tmps, op.ranges)
LinearAlgebra.ishermitian(::SectorOperator) = true
Base.eltype(::SectorOperator) = Float64

"""`v0`: optional starting vector for the iterative path -- e.g. the
eigenvector from a nearby coupling during a parameter scan, where the overlap
is 1 - O(dh^2) and convergence needs only a few matvecs."""
function eigenvalues(sector::Sector; k = 2, dense_limit = 4000, tol = 1e-9,
                     v0 = nothing)
    sector.dim == 0 && return Float64[]
    if sector.dim <= dense_limit
        matrix = dense(sector)
        return sort(eigvals(Symmetric(0.5 .* (matrix .+ matrix'))))[1:min(k, sector.dim)]
    end
    HAVE_ARPACK || error("Arpack.jl required for dim > $dense_limit")
    op = SectorOperator(sector, workspace(sector)...)
    kwargs = v0 === nothing ? (;) : (; v0 = v0 ./ norm(v0))
    values, _ = Arpack.eigs(op; nev = k, which = :SR, tol = tol,
                            ncv = min(sector.dim - 1, max(20, 20 * k)),
                            maxiter = 3000, kwargs...)
    return sort(real.(values))
end

"""Ground state and its vector, warm-startable: the scan workhorse."""
function ground_state_vector(sector::Sector; tol = 1e-9, v0 = nothing)
    if sector.dim <= 4000
        matrix = dense(sector)
        Fv = eigen(Symmetric(0.5 .* (matrix .+ matrix')))
        return Fv.values[1], Fv.vectors[:, 1]
    end
    op = SectorOperator(sector, workspace(sector)...)
    kwargs = v0 === nothing ? (;) : (; v0 = v0 ./ norm(v0))
    values, vectors = Arpack.eigs(op; nev = 1, which = :SR, tol = tol,
                                  ncv = min(sector.dim - 1, 20),
                                  maxiter = 3000, kwargs...)
    return real(values[1]), real.(vectors[:, 1])
end

"""Ground state only: plain three-term Lanczos, no reorthogonalisation.

Legitimate for k = 1 exactly because the failure mode of skipping
reorthogonalisation -- ghost copies of converged Ritz values -- cannot move
the minimum.  Cost per iteration is one matvec plus two axpys, against
ARPACK's full orthogonalisation of every new vector at large dim.
Convergence is declared when the lowest Ritz value stalls below `tol`
over a probe window; `validate` and the caller can cross-check against
`eigenvalues` on smaller sectors.
"""
function ground_state(sector::Sector; tol = 1e-9, maxiter = 2000)
    buffers, tmps, ranges = workspace(sector)
    n = sector.dim
    v = randn(Random.MersenneTwister(1), n)
    v ./= norm(v)
    v_old = zeros(n)
    w = zeros(n)
    alphas = Float64[]
    betas = Float64[]
    last = Inf
    for iteration in 1:maxiter
        matvec!(w, sector, v, buffers, tmps, ranges)
        alpha = dot(v, w)
        push!(alphas, alpha)
        @. w -= alpha * v
        if !isempty(betas)
            @. w -= betas[end] * v_old
        end
        beta = norm(w)
        beta < 1e-13 && break
        copyto!(v_old, v)
        @. v = w / beta
        push!(betas, beta)
        if iteration % 10 == 0 || iteration == maxiter
            T = SymTridiagonal(copy(alphas), copy(betas[1:end-1]))
            value = eigmin(T)
            abs(value - last) < tol * max(1.0, abs(value)) && return value
            last = value
        end
    end
    return eigmin(SymTridiagonal(alphas, betas[1:length(alphas)-1]))
end

"""Like `eigenvalues` but also returns eigenvectors (columns)."""
function eigensystem(sector::Sector; k = 2, dense_limit = 4000, tol = 1e-9)
    if sector.dim <= dense_limit
        matrix = dense(sector)
        F = eigen(Symmetric(0.5 .* (matrix .+ matrix')))
        order = sortperm(F.values)[1:min(k, sector.dim)]
        return F.values[order], F.vectors[:, order]
    end
    HAVE_ARPACK || error("Arpack.jl required for dim > $dense_limit")
    op = SectorOperator(sector, workspace(sector)...)
    values, vectors = Arpack.eigs(op; nev = k, which = :SR, tol = tol,
                                  ncv = min(sector.dim - 1, max(20, 20 * k)),
                                  maxiter = 3000)
    order = sortperm(real.(values))
    return real.(values)[order], real.(vectors)[:, order]
end

"""dE_i/dh by Hellmann-Feynman: the field term is -h (n_+ - n_-) and is
channel-diagonal, so the slope is -sum over channels of (N - 2 n_-) times
the eigenvector weight in that channel.  Validated in the dataset generator
against explicit re-solves at shifted h."""
function field_slopes(sector::Sector, vectors::Matrix{Float64})
    slopes = zeros(size(vectors, 2))
    for (n_minus, two_Jp, two_Jm, mult_p, mult_m) in sector.channels
        off = sector.offset[(n_minus, two_Jp, two_Jm)]
        span = off+1:off+mult_p*mult_m
        for i in axes(vectors, 2)
            slopes[i] -= (sector.N - 2 * n_minus) *
                         sum(abs2, @view vectors[span, i])
        end
    end
    return slopes
end

solve(N, h; V0 = 4.75, V1 = 1.0, L = 0, z2 = +1, k = 2,
      dense_limit = 4000, n_minus_max = nothing) =
    eigenvalues(Sector(N, h, V0, V1, L, z2; n_minus_max = n_minus_max);
                k = k, dense_limit = dense_limit)

# =============================================================================
# 6. Validation
# =============================================================================

# Reference energies cross-checked against an independent Python implementation
# (exact ED at N <= 8) and against FuzzifiED -- independent codes, agreeing to
# every digit shown.  h = 3.153.
const REFERENCE = Dict(
    (4, 0, 1) => [-8.068208], (6, 0, 1) => [-10.714531, -2.527597],
    (6, 0, -1) => [-7.633907, 3.785248], (6, 2, 1) => [7.052190, 8.416209],
    (8, 0, 1) => [-13.350083, -6.224854],
    (8, 0, -1) => [-10.689032, -0.692381],
    (8, 2, 1) => [1.992050, 3.597643],
    (12, 0, 1) => [-18.610298, -12.778193],
    (12, 0, -1) => [-16.446956, -8.224917],
    (12, 2, 1) => [-6.152416, -4.543555],
    (14, 0, 1) => [-21.238061, -15.838505],
    (14, 0, -1) => [-19.239030, -11.617298],
    (14, 2, 1) => [-9.727606, -8.176877],
    (16, 0, 1) => [-23.865039, -18.815716],
    (16, 0, -1) => [-21.998482, -14.864520],
    (16, 2, 1) => [-13.116024, -11.629255],
)

function mscheme_sector_dim(N, L, z2)
    orb = orbit(N)
    parity = z2 > 0 ? 0 : 1
    count(two_M_total) = sum(
        length(states) * block_dim(orb, n_minus, two_M_total - two_M)
        for n_minus in parity:2:N
        for (two_M, states) in blocks(orb, N - n_minus); init = 0)
    return count(2L) - count(2L + 2)
end

function validate()
    passes = fails = 0
    check(name, ok, detail = "") = begin
        ok ? (passes += 1) : (fails += 1)
        @printf("  [%s] %s   %s\n", ok ? "PASS" : "FAIL", name, detail)
    end

    println("1. orbit tables")
    for N in (6, 8)
        worst = maximum(abs(sum((J + 1) * m for (J, m) in multiplets(orbit(N), n);
                                init = 0) - binomial(N, n)) for n in 0:N)
        check("N = $N  sum (2J+1) mult = C(N,n)", worst == 0, "max |d| = $worst")
    end

    # dipole pseudopotentials V_J = J(J+1) are exactly solvable at h = 0:
    # E = 4 s(s+1) Nup Ndn + 2[L(L+1) - Lu(Lu+1) - Ld(Ld+1)] -- an analytic
    # end-to-end check of the whole coupled machinery at any N
    let N = 8, two_s = N - 1
        ps = [Float64((two_s - l) * (two_s - l + 1)) for l in 0:N-1]
        worst = 0.0
        for L in (0, 1, 2)
            ed = Float64[]
            for z2 in (1, -1)
                s = Sector(N, 0.0, ps, L, z2)
                s.dim > 0 && append!(ed, eigenvalues(s; k = s.dim,
                                                     dense_limit = 10^9))
            end
            sort!(ed)
            pred = Float64[]
            for nup in 0:N
                for (tLu, mu) in multiplets(orbit(N), nup),
                    (tLd, md) in multiplets(orbit(N), N - nup)
                    abs(tLu - tLd) <= 2L <= tLu + tLd || continue
                    e = two_s * (two_s + 2) * nup * (N - nup) +
                        2 * (L * (L + 1) - tLu * (tLu + 2) / 4 -
                             tLd * (tLd + 2) / 4)
                    for _ in 1:mu*md
                        push!(pred, e)
                    end
                end
            end
            sort!(pred)
            worst = length(ed) == length(pred) ?
                    max(worst, maximum(abs.(ed .- pred))) : Inf
        end
        check("dipole potential exactly solvable (N = 8, L = 0..2)",
              worst < 1e-9, @sprintf("max |dE| = %.1e", worst))
    end
    check("(7/2)^4 has 8 multiplets",
          sum(values(multiplets(orbit(8), 4))) == 8)

    println("2. the fitted scalar-product formula against explicit construction")
    worst = verify_scalar_formula()
    check("24 random tuples", worst < 1e-10, @sprintf("max |d| = %.1e", worst))

    println("3. sector dimensions against the m-scheme count")
    for N in (8, 12)
        worst = 0
        for (L, z2) in ((0, 1), (0, -1), (1, 1), (2, 1), (2, -1), (3, -1))
            sector = Sector(N, 3.153, 4.75, 1.0, L, z2)
            worst = max(worst, abs(sector.dim - mscheme_sector_dim(N, L, z2)))
        end
        check("N = $N  six sectors", worst == 0, "max |d dim| = $worst")
    end

    println("4. dense assembly == threaded matvec, and H symmetric")
    sector = Sector(12, 3.153, 4.75, 1.0, 2, 1)
    matrix = dense(sector)
    buffers, tmps, ranges = workspace(sector)
    rng = Random.MersenneTwister(0)
    worst = 0.0
    for _ in 1:3
        x = randn(rng, sector.dim)
        worst = max(worst, maximum(abs, matrix * x -
                                        matvec!(zeros(sector.dim), sector, x,
                                                buffers, tmps, ranges)))
    end
    check("N = 12 (2,+1) on random vectors", worst < 1e-9,
          @sprintf("max |d| = %.1e", worst))
    check("N = 12 (2,+1) symmetry", maximum(abs, matrix - matrix') < 1e-10)

    println("5. pseudopotential channels, one l at a time, against FuzzifiED")
    oracle = Dict(
        (6, [1.0], 1) => [-9.0, -3.0, -1.95238095],
        (6, [0.0, 1.0], 1) => [-2.63063152, -0.54574195, -0.34119421],
        (6, [0.0, 0.0, 1.0], -1) => [-6.0, -4.72222222, -4.54444444],
        (6, [0.0, 0.0, 0.0, 1.0], -1) => [-3.38904635, -3.15857942,
                                          -3.10425602],
        (6, [0.0, 0.0, 0.0, 0.0, 1.0], 1) => [-9.0, -3.0, -2.71428571],
        (8, [4.75, 1.0, 0.7, 0.3, 0.1], 1) => [-2.33015883, 3.55319357,
                                               8.32045468],
        (8, [4.75, 1.0, 0.7, 0.3, 0.1], -1) => [-2.30282461, 4.93598181,
                                                7.68194151])
    for (key, ref) in sort(collect(oracle); by = k -> (k[1][1], k[1][3]))
        (N, ps, z2) = key
        levels = Float64[]
        for L in 0:2*N
            sector = Sector(N, 1.5, Float64.(ps), L, z2)
            sector.dim == 0 && continue
            m = dense(sector)
            append!(levels, eigvals(Symmetric(0.5 .* (m .+ m'))))
        end
        ours = sort(levels)[1:length(ref)]
        worst = maximum(abs.(ours .- ref))
        check(@sprintf("N = %d ps_pot = %s z2 = %+d", N, string(ps), z2),
              worst < 3e-6, @sprintf("max |dE| = %.1e", worst))
    end

    println("6. thirty reference energies (independent Python ED + FuzzifiED)")
    for key in sort(collect(keys(REFERENCE)))
        N, L, z2 = key
        values = solve(N, 3.153; L = L, z2 = z2, k = length(REFERENCE[key]))
        worst = maximum(abs.(values[1:length(REFERENCE[key])] .- REFERENCE[key]))
        check(@sprintf("N = %2d (%d,%+d)", N, L, z2), worst < 3e-5,
              @sprintf("max |dE| = %.1e", worst))
    end

    @printf("\n%d/%d checks passed\n", passes, passes + fails)
    return fails == 0 ? 0 : 1
end

# =============================================================================
# 6b. Three-flavor O(2) extension (charge-resolved sectors)
# =============================================================================

include("O2Scheme.jl")

# =============================================================================
# 7. Command line
# =============================================================================

function main(args)
    if "--validate" in args
        exit(validate())
    end
    if "--validate-o2" in args
        exit(o2_validate())
    end
    N, h, k = 12, 3.153, 2
    for (i, a) in enumerate(args)
        a == "--N" && (N = parse(Int, args[i+1]))
        a == "--h" && (h = parse(Float64, args[i+1]))
        a == "--k" && (k = parse(Int, args[i+1]))
    end
    @printf("N = %d, h = %.3f, threads = %d\n\n", N, h, Threads.nthreads())
    @printf("  %10s %10s %8s %8s   levels\n", "sector", "dim", "build", "solve")
    energies = Dict{Tuple{Int,Int},Vector{Float64}}()
    for (L, z2) in ((0, 1), (0, -1), (2, 1))
        t0 = time()
        sector = Sector(N, h, 4.75, 1.0, L, z2)
        built = time() - t0
        t0 = time()
        values = eigenvalues(sector; k = k)
        elapsed = time() - t0
        energies[(L, z2)] = values
        @printf("  %10s %10s %7.1fs %7.1fs   %s\n",
                "($L,$(z2 > 0 ? "+" : "-")1)",
                string(sector.dim), built, elapsed,
                join([@sprintf("%.6f", v) for v in values], "  "))
    end
    E0 = energies[(0, 1)][1]
    gap_T = energies[(2, 1)][1] - E0
    @printf("\n  Delta_sigma = %.6f   (3D Ising: 0.518149)\n",
            3 * (energies[(0, -1)][1] - E0) / gap_T)
    length(energies[(0, 1)]) > 1 &&
        @printf("  Delta_epsilon = %.6f   (3D Ising: 1.412625)\n",
                3 * (energies[(0, 1)][2] - E0) / gap_T)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

end # module
