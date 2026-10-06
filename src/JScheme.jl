"""
    JScheme

Exact diagonalisation of the fuzzy-sphere Ising model in the J-scheme --
an exact-L alternative to its m-scheme ED at moderate and large N.
The basis is |[(n_+, J_+, a_+) x (n_-, J_-, a_-)] L>: good total angular
momentum L by construction and exact Z2 = (-1)^{n_-}

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
import SparseArrays

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
    for (i, c) in enumerate(_SIXJ_CACHES)
        merge!(_SIXJ_SHARED, c)
        _SIXJ_CACHES[i] = Dict{NTuple{6,Int},Float64}()   # empty! would keep the capacity
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

"""Racah 6j, memoised under the 24 symmetries.  The alternating sum can cancel
catastrophically in floating point at the j this file needs, so
`_sixj_compute` measures the cancellation and falls back to exact integer
arithmetic when it matters."""
function sixj(a1, a2, a3, a4, a5, a6)
    key = _sixj_canonical(a1, a2, a3, a4, a5, a6)
    shared = get(_SIXJ_SHARED, key, NaN)
    isnan(shared) || return shared
    _SIXJ_CACHE = _SIXJ_CACHES[Threads.threadid()]
    haskey(_SIXJ_CACHE, key) && return _SIXJ_CACHE[key]
    return _SIXJ_CACHE[key] = _sixj_compute(key...)
end

"""Racah sum without factorials.  Consecutive terms
    term(t) = (-1)^t (t+1)! / [prod (t - low)! prod (high - t)!]
have the ratio  r_t = -(t+2) prod(high - t) / prod(t+1 - low)  of small integers, so
    sum_t term(t) = term(tmin) [1 + r_tmin (1 + r_tmin+1 (1 + ...))],
with term(tmin) and the triangle prefactor from log-factorials.  The bracket is
evaluated in Float64 together with the same recursion on |r_t|, whose ratio to
the result bounds the cancellation; when that bound times the number of terms
exceeds 1e3 the bracket is recomputed exactly as one BigInt fraction (BigInt x
Int products only).  Against the exact Rational{BigInt} form
(`_sixj_compute_rational`, kept as the reference) the difference is at most
4e-15 absolute for 2j <= 120, and it is 35-55x faster; the rational form had
taken 90% of the first O(2) build at N = 12 (973k distinct symbols)."""
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
    tmin, tmax = maximum(lows), minimum(highs)
    tmin > tmax && return 0.0
    log_term = logfact(tmin + 1) - sum(logfact(tmin - l) for l in lows) -
               sum(logfact(h - tmin) for h in highs)
    sgn = isodd(tmin) ? -1.0 : 1.0
    # Float64 Horner, with the cancellation factor sum|terms| / |sum| alongside;
    # its rounding error is ~ (terms x factor) eps, accepted below 1e3 eps
    # (the exp of the log-factorials already carries ~1e-13)
    v, va = 1.0, 1.0
    for t in tmax-1:-1:tmin
        a = -(t + 2) * (highs[1] - t) * (highs[2] - t) * (highs[3] - t)
        b = (t + 1 - lows[1]) * (t + 1 - lows[2]) * (t + 1 - lows[3]) *
            (t + 1 - lows[4])
        r = a / b
        v = 1.0 + r * v
        va = 1.0 + abs(r) * va
    end
    if va * (tmax - tmin + 1) < 1e3 * abs(v)
        return sgn * sign(v) * exp(log_prefactor + log_term + log(abs(v)))
    end
    num, den = big(1), big(1)                    # exact: BigInt x Int only
    for t in tmax-1:-1:tmin
        a = -(t + 2) * (highs[1] - t) * (highs[2] - t) * (highs[3] - t)
        b = (t + 1 - lows[1]) * (t + 1 - lows[2]) * (t + 1 - lows[3]) *
            (t + 1 - lows[4])
        num = den * b + num * a
        den = den * b
    end
    iszero(num) && return 0.0
    return sgn * sign(num) * sign(den) *
           exp(log_prefactor + log_term + _logabs(num) - _logabs(den))
end

"log|x| of a BigInt through a 62-bit Float64 mantissa (no BigFloat)."
function _logabs(x::BigInt)
    s = max(0, ndigits(x; base = 2) - 62)
    return log(Float64(abs(x) >> s)) + s * log(2.0)
end

function _sixj_compute_rational(j1, j2, j3, j4, j5, j6)
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
    ph::Dict{NTuple{2,Int},Tuple{Vector{Int},Vector{Int}}}
end

Orbit(N) = Orbit(N, N - 1, Dict(), Dict(), Dict(), Dict(), Dict(), Dict(),
                 Dict())

const _ORBITS = Dict{Int,Orbit}()

"""The single-tower tables for N orbitals, loaded from the disk cache when one
is enabled and holds them (see `enable_cache!`)."""
function orbit(N)
    get!(_ORBITS, N) do
        orb = Orbit(N)
        CACHE_DIR[] === nothing || load_tables!(orb)
        orb
    end
end

# -----------------------------------------------------------------------------
# Disk cache of the coupling-independent tables
#
# Everything a sector build computes before it assembles tasks depends on N
# alone: the highest-weight bases, the reduced matrix elements, the pair
# scalars, the particle-hole maps and the 6j symbols.  At N = 17-18 these take
# 7-17 s, against 0.05-0.4 s for the assembly, and load from disk in well under
# a second.  One file per N plus one 6j file; opt-in, off by default.
#
# The bases come from SVD nullspaces, which may pick a different (equally
# valid) basis on another run, and every reduced matrix element refers to the
# basis it was computed in.  A small sidecar file holds a fingerprint of the
# bases; a session that loaded a file keeps using its bases, and writes back
# only if the file on disk still has the same fingerprint.  Files written by
# other code versions or Julia versions are ignored (`_CACHE_TAG`).
# -----------------------------------------------------------------------------

import Serialization

const CACHE_DIR = Ref{Union{Nothing,String}}(nothing)
const _CACHE_TAG = string("JScheme-tables-v2 julia-", VERSION, " src-",
    string(hash(read(joinpath(@__DIR__, "JScheme.jl"), String),
                hash(read(joinpath(@__DIR__, "O2Scheme.jl"), String))); base = 16))
const _SAVED = Dict{Int,Int}()           # N => table count at last load/save
const _SIXJ_LOADED = Ref(false)

"""    enable_cache!(dir = ~/.cache/FuzzySphereLDiag)

Keep the coupling-independent tables of every N on disk, so that later
sessions build sectors without recomputing them.  Also enabled at load time by
the environment variable `FUZZYSPHERELDIAG_CACHE=dir`.  A file takes about
0.3 GB at N = 17 and 0.6 GB at N = 18."""
function enable_cache!(dir::AbstractString = joinpath(homedir(), ".cache",
                                                      "FuzzySphereLDiag"))
    mkpath(dir)
    CACHE_DIR[] = abspath(dir)
    for orb in values(_ORBITS)         # tables already in memory keep their bases
        load_tables!(orb)
    end
    return CACHE_DIR[]
end

disable_cache!() = (CACHE_DIR[] = nothing; nothing)

function __init__()
    dir = get(ENV, "FUZZYSPHERELDIAG_CACHE", "")
    isempty(dir) || enable_cache!(dir)
end

_table_file(N) = joinpath(CACHE_DIR[], "tables_N$(N).jls")
_print_file(N) = joinpath(CACHE_DIR[], "tables_N$(N).fingerprint")
_sixj_file() = joinpath(CACHE_DIR[], "sixj.jls")
_table_count(orb::Orbit) = length(orb.hw) + length(orb.rme_t) +
    length(orb.rme_pair) + length(orb.rme_y) + length(orb.w) + length(orb.ph)

function _fingerprint(hw_tables)
    h = hash(:hw)
    for n in sort(collect(keys(hw_tables)))
        for (two_J, m) in sort(collect(hw_tables[n]); by = first)
            h = hash((n, two_J, size(m)), hash(m, h))
        end
    end
    return h
end

function _read_cache(file)
    isfile(file) || return nothing
    data = try
        open(Serialization.deserialize, file)
    catch err
        @warn "ignoring unreadable cache file" file exception = err
        return nothing
    end
    (data isa NamedTuple && get(data, :tag, "") == _CACHE_TAG) || return nothing
    return data
end

function _write_cache(file, data)
    tmp = file * ".tmp." * string(getpid())
    open(io -> Serialization.serialize(io, data), tmp, "w")
    mv(tmp, file; force = true)                # atomic on one filesystem
end

"""Fill an empty `Orbit` from the cache.  An orbit that already holds bases
is left alone: its matrix elements refer to them."""
function load_tables!(orb::Orbit)
    _load_sixj!()
    isempty(orb.hw) || return false
    data = _read_cache(_table_file(orb.N))
    (data === nothing || data.N != orb.N) && return false
    merge!(orb.hw, data.hw)
    merge!(orb.rme_t, data.rme_t)
    merge!(orb.rme_pair, data.rme_pair)
    merge!(orb.rme_y, data.rme_y)
    merge!(orb.w, data.w)
    merge!(orb.ph, data.ph)
    _SAVED[orb.N] = _table_count(orb)
    return true
end

function _load_sixj!()
    _SIXJ_LOADED[] && return
    _SIXJ_LOADED[] = true
    data = _read_cache(_sixj_file())
    data === nothing || merge!(_SIXJ_SHARED, data.sixj)
end

"""Write the tables of N to the cache if the session added any.  Called at
the end of every sector build when the cache is enabled; single-threaded."""
function save_tables(N)
    CACHE_DIR[] === nothing && return false
    orb = orbit(N)
    count = _table_count(orb)
    get(_SAVED, N, -1) == count && return false
    mine = string(_CACHE_TAG, " ", _fingerprint(orb.hw))
    theirs = isfile(_print_file(N)) ? read(_print_file(N), String) : ""
    if isfile(_table_file(N)) && startswith(theirs, _CACHE_TAG) && theirs != mine
        # another session wrote tables in different bases first: keep theirs
        return false
    end
    _write_cache(_table_file(N), (tag = _CACHE_TAG, N = N, hw = orb.hw,
        rme_t = orb.rme_t, rme_pair = orb.rme_pair, rme_y = orb.rme_y,
        w = orb.w, ph = orb.ph))
    tmp = _print_file(N) * ".tmp." * string(getpid())
    write(tmp, mine)
    mv(tmp, _print_file(N); force = true)
    merge_sixj_caches!()
    sixj = _read_cache(_sixj_file())           # 6j values do not depend on bases
    sixj === nothing || merge!(_SIXJ_SHARED, sixj.sixj)
    _write_cache(_sixj_file(), (tag = _CACHE_TAG, sixj = _SIXJ_SHARED))
    _SAVED[N] = count
    return true
end

"""Forget every in-memory table (orbits, 6j, CG, Pandya); for tests and timing."""
function reset_tables!()
    empty!(_ORBITS); empty!(_SAVED); empty!(_SIXJ_SHARED)
    foreach(empty!, _SIXJ_CACHES); empty!(_CG_CACHE)
    empty!(_PANDYA); empty!(_PAIRHOP)
    _SIXJ_LOADED[] = false
    return nothing
end

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

"""An operator (sum of fermion strings) between two M-blocks of one orbit, as
a sparse matrix: a one-body or pair term connects each determinant to at most
about N others, while the blocks reach several thousand determinants at
N = 20.  Built dense, the products with the highest-weight vectors in
`stretched` cost d_out/N times the necessary work, which made the reduced
matrix elements the dominant cold-build cost from N = 19 on.  Duplicate
entries are summed, as the dense accumulation did."""
function block_operator(orb::Orbit, terms, n_in, two_M_in, n_out, two_M_out)
    source = get(blocks(orb, n_in), two_M_in, Int[])
    target = get(blocks(orb, n_out), two_M_out, Int[])
    index = Dict(s => i for (i, s) in enumerate(target))
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for (column, state) in enumerate(source)
        for (coefficient, ops) in terms
            out, sign = apply_ops(state, ops)
            if sign != 0
                row = get(index, out, 0)
                if row != 0
                    push!(rows, row); push!(cols, column); push!(vals, sign * coefficient)
                end
            end
        end
    end
    return SparseArrays.sparse(rows, cols, vals, length(target), length(source))
end

"""Kernel of the raising operator J+ on one M-block, as the zero eigenspace of
J+' J+ = J- J+ = J^2 - M(M+1) restricted to the block.  Its spectrum,
J(J+1) - M(M+1) for J >= M, is known exactly, so zero is separated from the
rest by at least 2(M+1) >= 2, and a symmetric eigensolver restricted to
(-1, 1] returns just the kernel.  This replaces a full SVD of J+, whose
divide-and-conquer driver failed to converge on some blocks from N = 18 on and
fell back to QR iteration, single-threaded and taking minutes per block at
N = 20.  The kernel dimension is asserted to be dim(M) - dim(M+1)."""
function hw_kernel(raising::SparseArrays.SparseMatrixCSC{Float64,Int}, mult::Int)
    G = Symmetric(Matrix(raising' * raising))
    F = eigen(G, -1.0, 1.0)
    @assert length(F.values) == mult "kernel of J+ has $(length(F.values)) vectors, expected $mult"
    return Matrix(F.vectors)
end

"""A * B for sparse A and dense B, threaded over the columns of B when the
product is large; SparseArrays' own product runs on one thread."""
function _sparse_times_dense(A::SparseArrays.SparseMatrixCSC{Float64,Int}, B::AbstractMatrix{Float64})
    m = size(B, 2)
    C = zeros(size(A, 1), m)
    if m >= 2 && Threads.nthreads() > 1 && SparseArrays.nnz(A) * m > 2^18
        Threads.@threads for j in 1:m
            mul!(view(C, :, j), A, view(B, :, j))
        end
    else
        mul!(C, A, B)
    end
    return C
end

"""Particle-hole conjugation of one flavour tower, c+_m -> (-1)^(j-m) c_{-m},
the hole operator that transforms like c+_m, so the map commutes with
rotations and keeps M.  The canonical determinant c+_{a1}...c+_{an}|0>
(a1 < ... < an) goes to the image string applied to the full shell
|full> = c+_0...c+_{N-1}|0>.  For each state of blocks(orb, n)[two_M] returns
the position of its image in blocks(orb, N-n)[two_M] and the sign."""
function ph_image(orb::Orbit, n::Int, two_M::Int)
    source = get(blocks(orb, n), two_M, Int[])
    target = get(blocks(orb, orb.N - n), two_M, Int[])
    index = Dict(s => i for (i, s) in enumerate(target))
    full = (1 << orb.N) - 1
    rows = zeros(Int, length(source))
    signs = zeros(Int, length(source))
    for (column, state) in enumerate(source)
        ops = Tuple{Bool,Int}[]
        sign = 1
        bits = state
        while bits != 0
            k = trailing_zeros(bits)             # ascending: reading order
            push!(ops, (false, orb.N - 1 - k))   # m -> -m
            isodd(orb.two_j - k) && (sign = -sign)   # j - m = two_j - k
            bits &= bits - 1
        end
        out, s = apply_ops(full, ops)
        @assert s != 0 && haskey(index, out)
        rows[column] = index[out]
        signs[column] = sign * s
    end
    return rows, signs
end

"""Apply the tower particle-hole map to the columns of `vectors`, which live
on blocks(orb, n)[two_M]; the result lives on blocks(orb, N-n)[two_M]."""
function ph_apply(orb::Orbit, n::Int, two_M::Int, vectors::Matrix{Float64})
    rows, signs = ph_image(orb, n, two_M)
    out = zeros(block_dim(orb, orb.N - n, two_M), size(vectors, 2))
    for (r, row) in enumerate(rows)
        @views out[row, :] .= signs[r] .* vectors[r, :]
    end
    return out
end

"""Highest-weight vectors: {two_J => (d_block x mult)}, kernel of J+.

The bases are chosen so that the tower particle-hole map `ph_image` is a
signed permutation of multiplet labels (see `ph_multiplet_map`): for n > N/2
the vectors are the particle-hole images of those for N - n, and for the
self-conjugate n = N/2 the SVD basis is rotated to the real Schur form of the
map.  Any orthonormal choice gives the same spectra; this one lets the
Ising sectors be split by particle-hole parity (`Sector(...; ph = +-1)`)."""
function hw(orb::Orbit, n::Int)
    get!(orb.hw, n) do
        2n > orb.N && return Dict(two_J => ph_apply(orb, orb.N - n, two_J, m)
                                  for (two_J, m) in hw(orb, orb.N - n))
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
                out[two_M] = hw_kernel(raising, length(states) - d_up)
            end
        end
        if 2n == orb.N
            # self-conjugate tower: the particle-hole map U = W' C W is
            # orthogonal with U^2 = +-1; its real Schur form is diagonal (+-1)
            # or 2x2 rotations by pi/2, i.e. a signed permutation
            for (two_J, W) in out
                U = W' * ph_apply(orb, n, two_J, W)
                F = schur(U)
                out[two_J] = W * F.Z
            end
        end
        out
    end
end

multiplets(orb::Orbit, n) = Dict(J => size(m, 2) for (J, m) in hw(orb, n))

"""Particle-hole map on multiplet labels: C hw(n)[J] = hw(N-n)[J] U, with U a
signed permutation in the bases chosen by `hw`.  Returns (perm, sign): column
a of hw(n)[J] goes to column perm[a] of hw(N-n)[J] with sign[a].  Both the
invariance of the highest-weight space and the permutation structure are
asserted, so a wrong phase in `ph_image` cannot pass silently."""
function ph_multiplet_map(orb::Orbit, n::Int, two_J::Int)
    get!(orb.ph, (n, two_J)) do
        source = hw(orb, n)[two_J]
        target = hw(orb, orb.N - n)[two_J]
        image = ph_apply(orb, n, two_J, source)
        U = target' * image
        @assert maximum(abs, image .- target * U; init = 0.0) < 1e-9 "ph map leaves the highest-weight space"
        m = size(U, 2)
        perm = zeros(Int, m)
        sign = zeros(Int, m)
        for a in 1:m
            b = argmax(abs.(U[:, a]))
            @assert abs(abs(U[b, a]) - 1) < 1e-9 "ph map is not a signed permutation"
            perm[a] = b
            sign[a] = U[b, a] > 0 ? 1 : -1
        end
        @assert sort(perm) == 1:m
        (perm, sign)
    end
end

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
    matrix = hw_out' * _sparse_times_dense(op, hw_in)
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
    dst.ph == 0 || error("reduced_chi needs the full block; build dst with ph = 0")
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

"""All tasks that write to one output channel through the same right factor
R_+:  Y_out += (sum_t coef_t R_-^(t) X_in^(t)) R_+', the right product formed
once.  R_+ is cached per (n, lambda, J', J), so tasks that differ only in the
input J_- hold the same object and fall in one group.  On the L = 2 blocks
this cuts the arithmetic to 0.61 of task-by-task (grouping on the input side
instead gives 0.67); at L = 0 every group holds a single task."""
struct OutGroup
    rp::Matrix{Float64}                          # (mult'_+, mult_+)
    in_off::Vector{Int}
    coef::Vector{Float64}
    rm::Vector{Matrix{Float64}}                  # (mult'_-, mult_-) each
end

"""One output channel with its diagonal part and task groups.  Channels are
disjoint blocks of rows, so matvec workers own whole channels and write y
directly.  `diag` indexes `Sector.diagonal`; `cost` is in multiply-adds."""
struct OutChannel
    off::Int
    mp::Int                                      # mult'_+
    mm::Int                                      # mult'_-
    diag::Int
    has_w::Bool                                  # same-flavour W_+-, if nonzero
    groups::Vector{OutGroup}
    cost::Int
end

function build_work(tasks::Vector{Task}, diagonal)
    groups_at = Dict{Int,Vector{OutGroup}}()
    index = Dict{Tuple{Int,UInt},Int}()
    for t in tasks
        list = get!(() -> OutGroup[], groups_at, t.out_off)
        key = (t.out_off, objectid(t.rp))
        g = get(index, key, 0)
        if g == 0
            push!(list, OutGroup(t.rp, Int[], Float64[], Matrix{Float64}[]))
            g = index[key] = length(list)
        end
        push!(list[g].in_off, t.in_off)
        push!(list[g].coef, t.coef)
        push!(list[g].rm, t.rm)
    end
    work = OutChannel[]
    for (k, (off, scalar, w_plus, w_minus)) in enumerate(diagonal)
        mp, mm = size(w_plus, 1), size(w_minus, 1)
        has_w = !(iszero(w_plus) && iszero(w_minus))
        groups = get(groups_at, off, OutGroup[])
        cost = mp * mm * (has_w ? 1 + mp + mm : 1)
        for g in groups
            a, b = size(g.rp)
            @assert a == mp && all(size(r, 1) == mm for r in g.rm)
            cost += mm * b * (a + sum(size(r, 2) for r in g.rm))
        end
        push!(work, OutChannel(off, mp, mm, k, has_w, groups, cost))
        delete!(groups_at, off)
    end
    @assert isempty(groups_at) "task output outside the diagonal channels"
    sort!(work; by = w -> -w.cost)               # largest first for the queue
    return work
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
    work::Vector{OutChannel}  # the tasks by output channel, for the matvec
    ph::Int                   # 0: full block; +-1: particle-hole parity
    rep::Vector{Int}          # reduced basis vector k = (e_rep + coef e_partner)/sqrt2,
    partner::Vector{Int}      #   or e_rep alone where partner = 0
    coef::Vector{Float64}
end

"Dimension of the block actually diagonalised (the particle-hole half if ph != 0)."
rdim(s::Sector) = s.ph == 0 ? s.dim : length(s.rep)

Sector(N, h, V0::Real, V1::Real, L, z2; kwargs...) =
    Sector(N, h, [Float64(V0), Float64(V1)], L, z2; kwargs...)

"""General pseudopotential list: ps_pot[l+1] = V_l, FuzzifiED convention
(their `GetDenIntTerms` argument is `2 .* ps_pot`).  In the sigma^x basis the
even-l potentials feed the (+-) channels at odd pair-J = N-1-l and the odd-l
potentials feed the same-flavour channels at even pair-J -- the structure that
`validate` pins against FuzzifiED one l at a time.

`ph = +-1` restricts the block to one parity of the particle-hole symmetry
(see `ph_permutation`), which commutes with rotations and halves the block;
`ph = 0` keeps the whole (L, Z2) block.  Only the tasks whose output lies in
the channels with J_+ >= J_- are built, since a vector of definite parity is
fixed by those."""
function Sector(N, h, ps_pot::Vector{Float64}, L, z2; n_minus_max = nothing,
                ph = 0)
    ph in (-1, 0, 1) || error("ph must be -1, 0 or +1")
    keep(ch) = ph == 0 || ch[2] >= ch[3]
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
        keep((n_minus, two_Jp, two_Jm)) || continue
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
            keep(out_ch) || continue
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
                    keep(out_ch) || continue
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
    rep, partner, coef = Int[], Int[], Float64[]
    if ph != 0
        perm, sign = ph_permutation(N, two_L, channels, offset, position)
        for (n_minus, two_Jp, two_Jm, mult_p, mult_m) in channels
            two_Jp >= two_Jm || continue
            off = offset[(n_minus, two_Jp, two_Jm)]
            for i in off+1:off+mult_p*mult_m
                j = perm[i]
                if j == i
                    sign[i] == ph && (push!(rep, i); push!(partner, 0);
                                      push!(coef, 0.0))
                elseif two_Jp > two_Jm || i < j
                    push!(rep, i); push!(partner, j); push!(coef, ph * sign[i])
                end
            end
        end
    end
    CACHE_DIR[] === nothing || save_tables(N)
    return Sector(N, h, L, z2, position, channels, offset, diagonal, tasks,
                  build_work(tasks, diagonal), ph, rep, partner, coef)
end

"""Particle-hole symmetry of the Ising sector as a signed permutation of the
coupled basis: basis vector i goes to sign[i] * e_perm[i].

The symmetry swaps the two sigma^x towers and conjugates each one
(`ph_image`), so channel (n_-, J_+, J_-) goes to (n_-, J_-, J_+) with the
multiplet labels permuted by `ph_multiplet_map`.  The channel sign has two
parts, (-1)^(n_+ (n_- + N)) from carrying the image of the + string past the
- string and the filled shell, and (-1)^(J_+ + J_- - L) from recoupling in
the opposite order.  These give a symmetry for either overall sign; the
factor z2 (-1)^(N(N+1)/2) = (-1)^n_- (-1)^(N(N+1)/2) makes it equal to
FuzzifiED's GetParityQNOffd(N, 2, [2, 1], [-1, 1]) * GetRotyQNOffd(N, 2), so
ph = +1 is FuzzifiED's P R_y = +1 sector.  The (-1)^n_- is FuzzifiED's minus
sign on the second flavour; the N-dependent part was FITTED against
FuzzifiED at N = 6..11 and is checked in `validate` -- validated, not
derived.  [H, P] = 0 and P^2 = 1 are checked there too."""
function ph_permutation(N, two_L, channels, offset, dim)
    orb = orbit(N)
    perm = zeros(Int, dim)
    sign = zeros(Int, dim)
    for (n_minus, two_Jp, two_Jm, mult_p, mult_m) in channels
        n_plus = N - n_minus
        off = offset[(n_minus, two_Jp, two_Jm)]
        off_bar = offset[(n_minus, two_Jm, two_Jp)]
        s_c = phase(n_plus * (n_minus + N)) *
              phase((two_Jp + two_Jm - two_L) ÷ 2) *
              phase(n_minus) * phase(N * (N + 1) ÷ 2)       # FuzzifiED's label
        perm_p, sign_p = ph_multiplet_map(orb, n_plus, two_Jp)
        perm_m, sign_m = ph_multiplet_map(orb, n_minus, two_Jm)
        for a in 1:mult_p, b in 1:mult_m
            i = off + (a - 1) * mult_m + b
            perm[i] = off_bar + (perm_m[b] - 1) * mult_p + perm_p[a]
            sign[i] = Int(s_c) * sign_p[a] * sign_m[b]
        end
    end
    return perm, sign
end

# Small-GEMM microkernel.  The blocks are small (94% of the arithmetic has
# 16-64 rows and a 16-64 inner dimension), where a BLAS call costs more in
# dispatch than in arithmetic and OpenBLAS contends on its pool when called
# from many Julia threads.  The kernel keeps an 8x4 tile of C in registers and
# per step loads 8 elements of A and 4 of B for 32 multiply-adds; the earlier
# axpy loops loaded and stored C for every multiply-add.  Single-threaded on
# the M4 Max this runs at 22-28 G multiply-adds/s against 5-9 for the axpy
# form and 7-27 for OpenBLAS on the same shapes.
using Base.Cartesian: @nexprs

for (MR, NR) in ((8, 4), (4, 4), (2, 4), (1, 4), (8, 1), (4, 1), (2, 1), (1, 1))
    name = Symbol(:_tile_, MR, :x, NR, :!)
    @eval @inline function $name(C, cb, ldc, A, ab, lda, B, bb, bsl, bsj, k, alpha)
        @nexprs $NR j -> @nexprs $MR i -> c_i_j = 0.0
        @inbounds for l in 0:k-1
            ao = ab + l * lda
            @nexprs $MR i -> a_i = A[ao+i]
            bo = bb + l * bsl
            @nexprs $NR j -> b_j = B[bo+(j-1)*bsj+1]
            @nexprs $NR j -> @nexprs $MR i -> c_i_j = muladd(a_i, b_j, c_i_j)
        end
        @inbounds @nexprs $NR j -> @nexprs $MR i -> C[cb+(j-1)*ldc+i] += alpha * c_i_j
        return nothing
    end
end

"""C[coff + (j-1) m + i] += alpha sum_l A[aoff + (l-1) lda + i] B[boff + (l-1) bsl + (j-1) bsj + 1]
for i <= m, j <= n, l <= k: C is m x n column-major in a flat array, A is read
column-major with leading dimension lda, B through arbitrary strides (bsl, bsj),
so B can be a column-major block (1, k) or the transpose of a matrix (n, 1)."""
function _gemm!(C, coff, m, n, A, aoff, lda, B, boff, bsl, bsj, k, alpha)
    j0 = 0
    while j0 + 4 <= n
        cb = coff + j0 * m
        bb = boff + j0 * bsj
        i0 = 0
        while i0 + 8 <= m
            _tile_8x4!(C, cb + i0, m, A, aoff + i0, lda, B, bb, bsl, bsj, k, alpha)
            i0 += 8
        end
        i0 + 4 <= m && (_tile_4x4!(C, cb + i0, m, A, aoff + i0, lda, B, bb, bsl, bsj, k, alpha); i0 += 4)
        i0 + 2 <= m && (_tile_2x4!(C, cb + i0, m, A, aoff + i0, lda, B, bb, bsl, bsj, k, alpha); i0 += 2)
        i0 + 1 <= m && _tile_1x4!(C, cb + i0, m, A, aoff + i0, lda, B, bb, bsl, bsj, k, alpha)
        j0 += 4
    end
    while j0 < n
        cb = coff + j0 * m
        bb = boff + j0 * bsj
        i0 = 0
        while i0 + 8 <= m
            _tile_8x1!(C, cb + i0, m, A, aoff + i0, lda, B, bb, bsl, bsj, k, alpha)
            i0 += 8
        end
        i0 + 4 <= m && (_tile_4x1!(C, cb + i0, m, A, aoff + i0, lda, B, bb, bsl, bsj, k, alpha); i0 += 4)
        i0 + 2 <= m && (_tile_2x1!(C, cb + i0, m, A, aoff + i0, lda, B, bb, bsl, bsj, k, alpha); i0 += 2)
        i0 + 1 <= m && _tile_1x1!(C, cb + i0, m, A, aoff + i0, lda, B, bb, bsl, bsj, k, alpha)
        j0 += 1
    end
    return nothing
end

"""Threaded matvec over output channels.  Each channel is a disjoint block of
rows, so workers take whole channels from a shared counter, largest first,
and write y directly: no per-thread copies of y, no reduction, and a slower
core simply ends up with fewer channels.  `buffers` and `ranges` are unused
and kept for the signature."""
function matvec!(y::AbstractVector{Float64}, sector::Sector,
                 x::AbstractVector{Float64}, buffers, tmps, ranges)
    work = sector.work
    next = Threads.Atomic{Int}(1)
    @sync for w in eachindex(tmps)
        Threads.@spawn begin
            tmp = tmps[w]
            while true
                k = Threads.atomic_add!(next, 1)
                k > length(work) && break
                _apply_channel!(y, sector, work[k], x, tmp)
            end
        end
    end
    return y
end

function _apply_channel!(y, sector::Sector, ch::OutChannel, x, tmp)
    off, mp, mm = ch.off, ch.mp, ch.mm
    _, scalar, w_plus, w_minus = sector.diagonal[ch.diag]
    @inbounds for i in off+1:off+mp*mm
        y[i] = scalar * x[i]
    end
    if ch.has_w
        _gemm!(y, off, mm, mp, w_minus, 0, mm, x, off, 1, mm, mm, 1.0)   # W_- X
        _gemm!(y, off, mm, mp, x, off, mm, w_plus, 0, mp, 1, mp, 1.0)    # X W_+' (W_+ symmetric)
    end
    for g in ch.groups
        b = size(g.rp, 2)
        @inbounds for i in 1:mm*b
            tmp[i] = 0.0
        end
        for t in eachindex(g.in_off)                                     # U += coef R_- X_in
            rm = g.rm[t]
            d = size(rm, 2)
            _gemm!(tmp, 0, mm, b, rm, 0, mm, x, g.in_off[t], 1, d, d, g.coef[t])
        end
        _gemm!(y, off, mm, mp, tmp, 0, mm, g.rp, 0, mp, 1, b, 1.0)       # Y += U R_+'
    end
    return nothing
end

function workspace(sector::Sector)
    nworkers = max(1, Threads.nthreads())
    max_tmp = 1
    for ch in sector.work, g in ch.groups
        max_tmp = max(max_tmp, ch.mm * size(g.rp, 2))
    end
    return (Vector{Float64}[], [zeros(max_tmp) for _ in 1:nworkers],
            UnitRange{Int}[])
end

function dense(sector::Sector)
    sector.ph == 0 || error("dense() needs the full block; build it with ph = 0")
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

# ---- particle-hole halves (Sector(...; ph = +-1)) ---------------------------

"""Reduced vector -> full coupled-basis vector of definite particle-hole parity."""
function ph_expand!(x::AbstractVector{Float64}, s::Sector, xr::AbstractVector{Float64})
    fill!(x, 0.0)
    r = inv(sqrt(2.0))
    @inbounds for k in eachindex(s.rep)
        if s.partner[k] == 0
            x[s.rep[k]] = xr[k]
        else
            x[s.rep[k]] = r * xr[k]
            x[s.partner[k]] = s.coef[k] * r * xr[k]
        end
    end
    return x
end

"""Full vector -> reduced coordinates by orthogonal projection."""
function ph_project!(xr::AbstractVector{Float64}, s::Sector, x::AbstractVector{Float64})
    r = inv(sqrt(2.0))
    @inbounds for k in eachindex(s.rep)
        p = s.partner[k]
        xr[k] = p == 0 ? x[s.rep[k]] : r * (x[s.rep[k]] + s.coef[k] * x[p])
    end
    return xr
end

"""H on the particle-hole half.  The input is expanded to a full vector of
definite parity; H x then has the same parity, so its reduced coordinates
follow from the representative entries alone, which are the only rows the
restricted task list computes."""
struct PHOperator <: AbstractMatrix{Float64}
    sector::Sector
    buffers::Vector{Vector{Float64}}
    tmps::Vector{Vector{Float64}}
    ranges::Vector{UnitRange{Int}}
    x::Vector{Float64}
    y::Vector{Float64}
end

PHOperator(s::Sector) = PHOperator(s, workspace(s)..., zeros(s.dim), zeros(s.dim))
Base.size(op::PHOperator) = (rdim(op.sector), rdim(op.sector))
Base.size(op::PHOperator, i::Int) = rdim(op.sector)
LinearAlgebra.ishermitian(::PHOperator) = true
Base.eltype(::PHOperator) = Float64

function LinearAlgebra.mul!(yr::AbstractVector, op::PHOperator, xr::AbstractVector)
    s = op.sector
    ph_expand!(op.x, s, xr)
    matvec!(op.y, s, op.x, op.buffers, op.tmps, op.ranges)
    q = sqrt(2.0)
    @inbounds for k in eachindex(s.rep)
        yr[k] = s.partner[k] == 0 ? op.y[s.rep[k]] : q * op.y[s.rep[k]]
    end
    return yr
end

"""Diagonal of H in the coupled basis, on the rows of the output channels:
the channel's diagonal part plus every task whose input channel is its
output channel, whose block coef R_+ (x) R_- contributes coef R_-[a,a] R_+[b,b]."""
function hdiag(s::Sector)
    d = zeros(s.dim)
    for ch in s.work
        off, mp, mm = ch.off, ch.mp, ch.mm
        _, scalar, w_plus, w_minus = s.diagonal[ch.diag]
        for ap in 1:mp, am in 1:mm
            d[off+(ap-1)*mm+am] = scalar +
                (ch.has_w ? w_minus[am, am] + w_plus[ap, ap] : 0.0)
        end
        for g in ch.groups, t in eachindex(g.in_off)
            g.in_off[t] == off || continue
            rm = g.rm[t]
            for ap in 1:mp, am in 1:mm
                d[off+(ap-1)*mm+am] += g.coef[t] * rm[am, am] * g.rp[ap, ap]
            end
        end
    end
    return d
end

"""Orthonormalise t against the first m columns of V (block Gram-Schmidt, two
passes) and store it as column m+1; returns the new size.  A separate function
rather than a closure, so that m is never captured and boxed.  (The same
trap cost 30 ms per correction at N = 17 when a convergence comprehension
captured theta, which is why theta is updated in place.)"""
function _dav_add!(V, h, t, m)
    # normalize first, so the linear-dependence cut below is relative: near
    # convergence at a tight tolerance the Olsen correction itself is ~1e-10,
    # and an absolute cut on it dropped every new vector and stalled the
    # iteration until maxiter
    t0 = norm(t)
    t0 == 0.0 && return m
    t ./= t0
    for _ in 1:2
        m == 0 && break
        Vm = view(V, :, 1:m)
        mul!(view(h, 1:m), Vm', t)
        mul!(t, Vm, view(h, 1:m), -1.0, 1.0)
    end
    nt = norm(t)
    nt < 1e-10 && return m
    V[:, m+1] .= t ./ nt
    return m + 1
end

"""Block Davidson for the k lowest eigenpairs of a symmetric operator, with the
diagonal `dg` as preconditioner and Olsen's correction
    t = -(D - theta)^-1 (r - eps x),  eps = x'(D - theta)^-1 r / x'(D - theta)^-1 x,
which keeps t from collapsing onto the Ritz vector x when D is a good
approximation.  Converged when every residual satisfies |r| <= tol |theta|,
ARPACK's criterion.  The subspace restarts from its `keep` lowest Ritz vectors
when it reaches `max_dim`; the projected matrix is extended column by column
and the subspace is never copied, so the cost beyond the matvecs is a few
passes over n x max_dim numbers per iteration.  Returns (values, vectors,
matvec count)."""
function davidson(op, dg::AbstractVector{Float64}; k = 2, tol = 1e-8,
                  maxiter = 2000, max_dim = max(16k, 32), keep = max(4k, 8),
                  v0 = nothing)
    n = size(op, 1)
    k = min(k, n)
    max_dim = min(max_dim, n)
    keep = clamp(keep, k, max_dim - k)
    V = zeros(n, max_dim)
    W = zeros(n, max_dim)
    T = zeros(max_dim, max_dim)
    X = zeros(n, k)
    R = zeros(n, k)
    t = zeros(n)
    h = zeros(max_dim)
    m = 0
    v0 === nothing || (t .= v0; m = _dav_add!(V, h, t, m))
    for i in partialsortperm(dg, 1:min(n, k + 1))       # lowest diagonal entries
        m < k && (fill!(t, 0.0); t[i] = 1.0; m = _dav_add!(V, h, t, m))
    end
    while m < k
        t .= randn(n)
        m = _dav_add!(V, h, t, m)
    end
    done = 0                                # columns of W and T already computed
    nmult = 0
    theta = zeros(k)
    converged = falses(k)
    for _ in 1:maxiter
        for c in done+1:m
            mul!(view(W, :, c), op, view(V, :, c))
            nmult += 1
            mul!(view(T, 1:c, c), view(V, :, 1:c)', view(W, :, c))
            T[c, 1:c-1] .= view(T, 1:c-1, c)
        end
        done = m
        F = eigen(Symmetric(T[1:m, 1:m]))
        theta .= view(F.values, 1:k)             # in place: never rebind theta
        S = F.vectors[:, 1:k]
        mul!(X, view(V, :, 1:m), S)
        mul!(R, view(W, :, 1:m), S)
        for j in 1:k
            @views R[:, j] .-= theta[j] .* X[:, j]
        end
        for j in 1:k                             # a loop, not a comprehension:
            converged[j] = norm(view(R, :, j)) <= tol * max(abs(theta[j]), 1.0)
        end                                      # closures box what they capture
        all(converged) && return theta, copy(X), nmult
        if m + count(!, converged) > max_dim             # thick restart
            Q = F.vectors[:, 1:keep]
            V[:, 1:keep] .= view(V, :, 1:m) * Q
            W[:, 1:keep] .= view(W, :, 1:m) * Q
            fill!(T, 0.0)
            for c in 1:keep
                T[c, c] = F.values[c]
            end
            m = done = keep
        end
        for j in 1:k
            converged[j] && continue
            th = theta[j]
            num = 0.0
            den = 0.0
            @inbounds for i in 1:n
                d = dg[i] - th
                d = abs(d) < 1e-6 ? copysign(1e-6, d) : d
                num += X[i, j] * R[i, j] / d
                den += X[i, j] * X[i, j] / d
            end
            eps = num / den
            @inbounds for i in 1:n
                d = dg[i] - th
                d = abs(d) < 1e-6 ? copysign(1e-6, d) : d
                t[i] = -(R[i, j] - eps * X[i, j]) / d
            end
            m = _dav_add!(V, h, t, m)
        end
    end
    @warn "Davidson did not converge" maxiter
    return theta, copy(X), nmult
end

"""Lowest k levels of a particle-hole half, with eigenvectors expanded to the
full coupled basis (so `field_slopes`, `apply_odd_tensor` and the rest take
them unchanged).  Dense below `dense_limit`, from rdim matvecs."""
function ph_eigensystem(s::Sector; k = 2, dense_limit = 256, tol = 1e-9,
                        v0 = nothing, want_vectors = true, solver = :davidson)
    n = rdim(s)
    n == 0 && return Float64[], zeros(s.dim, 0)
    op = PHOperator(s)
    if n <= dense_limit
        M = zeros(n, n)
        e = zeros(n)
        for c in 1:n
            e[c] = 1.0
            mul!(view(M, :, c), op, e)
            e[c] = 0.0
        end
        F = eigen(Symmetric(0.5 .* (M .+ M')))
        values, reduced = F.values[1:min(k, n)], F.vectors[:, 1:min(k, n)]
    elseif solver == :davidson
        start = v0 === nothing ? nothing : ph_project!(zeros(n), s, v0)
        values, reduced, _ = davidson(op, hdiag(s)[s.rep]; k = k, tol = tol, v0 = start)
    else
        HAVE_ARPACK || error("Arpack.jl required for rdim > $dense_limit")
        kwargs = v0 === nothing ? (;) :
                 (; v0 = (w = ph_project!(zeros(n), s, v0); w ./ norm(w)))
        vals, vecs = Arpack.eigs(op; nev = k, which = :SR, tol = tol,
                                 ncv = min(n - 1, max(20, 20 * k)),
                                 maxiter = 3000, kwargs...)
        order = sortperm(real.(vals))
        values, reduced = real.(vals)[order], real.(vecs)[:, order]
    end
    want_vectors || return values, zeros(s.dim, 0)
    full = zeros(s.dim, length(values))
    for c in axes(reduced, 2)
        ph_expand!(view(full, :, c), s, view(reduced, :, c))
    end
    return values, full
end

"""`v0`: optional starting vector for the iterative path -- e.g. the
eigenvector from a nearby coupling during a parameter scan, where the overlap
is 1 - O(dh^2) and convergence needs only a few matvecs.

`solver = :davidson` (default) or `:arpack`.  Both stop at |r| <= tol |E|.
Davidson with the diagonal of H as preconditioner needs 40-60 matvecs for two
levels at N = 16-17 where ARPACK (ncv = 40) needs 77-78, and its bookkeeping
costs about as much as ARPACK's, so it is 1.35-1.8x faster there."""
function eigenvalues(sector::Sector; k = 2, dense_limit = 4000, tol = 1e-9,
                     v0 = nothing, solver = :davidson)
    sector.ph == 0 ||
        return ph_eigensystem(sector; k = k, dense_limit = min(dense_limit, 256),
                              tol = tol, v0 = v0, want_vectors = false,
                              solver = solver)[1]
    sector.dim == 0 && return Float64[]
    if sector.dim <= dense_limit
        matrix = dense(sector)
        return sort(eigvals(Symmetric(0.5 .* (matrix .+ matrix'))))[1:min(k, sector.dim)]
    end
    if solver == :davidson
        op = SectorOperator(sector, workspace(sector)...)
        return davidson(op, hdiag(sector); k = k, tol = tol, v0 = v0)[1]
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
    sector.ph == 0 || return (r = ph_eigensystem(sector; k = 1, tol = tol, v0 = v0);
                              (r[1][1], r[2][:, 1]))
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
    sector.ph == 0 || return ph_eigensystem(sector; k = 1, tol = tol,
                                            want_vectors = false)[1][1]
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
function eigensystem(sector::Sector; k = 2, dense_limit = 4000, tol = 1e-9,
                     solver = :davidson)
    sector.ph == 0 ||
        return ph_eigensystem(sector; k = k, dense_limit = min(dense_limit, 256),
                              tol = tol, solver = solver)
    if sector.dim > dense_limit && solver == :davidson
        op = SectorOperator(sector, workspace(sector)...)
        values, vectors, _ = davidson(op, hdiag(sector); k = k, tol = tol)
        return values, vectors
    end
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
      dense_limit = 4000, n_minus_max = nothing, ph = 0) =
    eigenvalues(Sector(N, h, V0, V1, L, z2; n_minus_max = n_minus_max, ph = ph);
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

    println("7. particle-hole halves (Sector(...; ph = +-1))")
    let ps = [4.75, 1.0, 0.7, 0.3], worst_c = 0.0, worst_s = 0.0, worst_u = 0.0
        for N in (8, 9), L in (0, 1, 2, 3), z2 in (1, -1)
            full = Sector(N, 3.153, ps, L, z2)
            full.dim == 0 && continue
            perm, sign = ph_permutation(N, 2L, full.channels, full.offset, full.dim)
            P = zeros(full.dim, full.dim)
            for i in 1:full.dim
                P[perm[i], i] = sign[i]
            end
            H = dense(full)
            worst_c = max(worst_c, maximum(abs, P * H - H * P))
            worst_s = max(worst_s, maximum(abs, P * P - I))
            levels = Float64[]
            for ph in (1, -1)
                half = Sector(N, 3.153, ps, L, z2; ph = ph)
                rdim(half) > 0 && append!(levels,
                    ph_eigensystem(half; k = rdim(half), dense_limit = 10^6,
                                   want_vectors = false)[1])
            end
            e = sort(eigvals(Symmetric(0.5 .* (H .+ H'))))
            worst_u = max(worst_u, length(levels) == length(e) ?
                                   maximum(abs.(sort(levels) .- e)) : Inf)
        end
        check("N = 8, 9  [H, P] = 0", worst_c < 1e-10, @sprintf("max = %.1e", worst_c))
        check("N = 8, 9  P^2 = 1", worst_s == 0.0, @sprintf("max = %.1e", worst_s))
        check("N = 8, 9  the two halves give every level of the block",
              worst_u < 1e-9, @sprintf("max |dE| = %.1e", worst_u))
    end
    # FuzzifiED, L_z = 2, sectors (Z2, P*R_y) with GetParityQNOffd(N,2,[2,1],[-1,1])
    # * GetRotyQNOffd(N,2); lowest four levels (L >= 2 mixed), h = 3.153, V = (4.75, 1)
    fz = Dict((8, 1, 1) => [1.992050, 3.597643, 9.373894, 10.390592],
              (8, 1, -1) => [6.624008, 7.580421, 7.765778, 10.653743],
              (8, -1, 1) => [-0.346024, 6.712522, 8.086043, 8.108483],
              (8, -1, -1) => [4.072295, 7.796240, 8.133194, 10.039322],
              (9, 1, 1) => [-0.224191, 1.414438, 7.387467, 8.263813],
              (9, 1, -1) => [4.333926, 5.183054, 5.359007, 8.947664],
              (9, -1, 1) => [-2.383175, 4.911266, 5.587902, 6.127558],
              (9, -1, -1) => [2.015587, 6.295142, 6.865162, 7.382550])
    # In two of the eight halves the lowest L = 2 level lies above the four
    # FuzzifiED levels listed (L = 3, 4 fill the bottom of L_z = 2), so six
    # levels must be found under their own label and none under the other.
    let same = 0, other = 0
        for ((N, z2, ph), ref) in fz
            e0 = eigenvalues(Sector(N, 3.153, 4.75, 1.0, 2, z2; ph = ph); k = 1)[1]
            same += any(abs.(ref .- e0) .< 1e-5)
            other += any(abs.(fz[(N, z2, -ph)] .- e0) .< 1e-5)
        end
        check("parity labels == FuzzifiED P R_y (N = 8, 9, L = 2)",
              same == 6 && other == 0,
              "$same/6 under the same label, $other under the opposite one")
    end
    for key in sort(collect(keys(REFERENCE)))
        N, L, z2 = key
        N >= 12 || continue
        values = solve(N, 3.153; L = L, z2 = z2, k = 2, dense_limit = 64, ph = 1)
        worst = maximum(abs.(values[1:length(REFERENCE[key])] .- REFERENCE[key]))
        check(@sprintf("N = %2d (%d,%+d)  ph = +1 half", N, L, z2), worst < 3e-5,
              @sprintf("max |dE| = %.1e", worst))
    end

    println("8. Davidson eigensolver against ARPACK")
    let worst = 0.0, worst_d = 0.0
        for (N, L, z2, ph) in ((12, 0, 1, 0), (12, 2, -1, 0), (14, 0, 1, 1), (14, 2, 1, 1),
                               (14, 1, -1, -1))
            s = Sector(N, 3.153, 4.75, 1.0, L, z2; ph = ph)
            a = eigenvalues(s; k = 3, dense_limit = 64, tol = 1e-10, solver = :arpack)
            d = eigenvalues(s; k = 3, dense_limit = 64, tol = 1e-10, solver = :davidson)
            worst = max(worst, maximum(abs.(a .- d)))
        end
        s = Sector(10, 3.153, [4.75, 1.0, 0.7, 0.3], 2, 1)
        worst_d = maximum(abs, hdiag(s) .- diag(dense(s)))
        check("diagonal of H == dense diagonal (N = 10)", worst_d < 1e-12,
              @sprintf("max |d| = %.1e", worst_d))
        check("three lowest levels, five sectors, N = 12-14", worst < 1e-8,
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
