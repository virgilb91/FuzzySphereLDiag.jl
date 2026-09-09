# =============================================================================
# O2Scheme.jl -- exact-L, exact-Q diagonalization of the spin-1 (three-flavor)
# fuzzy-sphere O(2) model of arXiv:2604.18705:
#
#   H = H00 + Hxy + HD,   flavors sigma = +1, 0, -1,   nu = 1/3 (Norb = N),
#   H00 = sum U^{m1m2m3m4} (c+_{m1} . c_{m4})(c+_{m2} . c_{m3})    [flavor .]
#   Hxy = -1/2 sum U [(c+ Sx c)(c+ Sx c) + (c+ Sy c)(c+ Sy c)]
#   HD  = D sum_m c+_m (Sz)^2 c_m,
#   U^{m1m2m3m4} = sum_l V_l C(s m1, s m2|J M) C(s m4, s m3|J M),  J = 2s - l.
#
# Conventions pinned by measurement, not by assertion: the literal bilinear-product
# Hamiltonian equals the normal-ordered one at anisotropy D + kappa/2 with
#   kappa = sum_l (-1)^l V_l (2J+1)/(2s+1),
# verified exactly against FuzzifiED at N = 6, 7.
#
# Basis chain per sector (N, L, Q):  blocks (n_+, n_-), n_0 = N - n_+ - n_-,
#   | [ (J_+ J_-) Jc , J_0 ] L >,   amplitudes A[a_-, a_+, a_0] (minus fastest).
# All channel weights are numerically projected from m-scheme coefficient
# tensors with asserted residuals; all recoupling phase conventions are
# verified against explicit Clebsch-Gordan constructions.  House rule: nothing
# hand-derived is trusted untested.
# =============================================================================

# expects JScheme.jl to be included first (Orbit machinery, cg/sixj, RMEs)

o2_kernel_kappa(N, ps) = sum((-1)^l * ps[l+1] * (2 * (N - 1 - l) + 1) / N
                             for l in 0:length(ps)-1; init = 0.0)

"""U^{m1m2m3m4} as a dense array over orbital indices k = (m + s) in 1..N,
index order [k1, k2, k3, k4]."""
function o2_kernel(N, ps)
    two_j = N - 1
    U = zeros(N, N, N, N)
    for l in 0:length(ps)-1
        V = ps[l+1]
        V == 0.0 && continue
        two_J = 2 * (two_j - l)
        for k1 in 1:N, k2 in 1:N
            two_M = (2k1 - 2 - two_j) + (2k2 - 2 - two_j)
            abs(two_M) > two_J && continue
            c12 = cg(two_j, 2k1 - 2 - two_j, two_j, 2k2 - 2 - two_j,
                     two_J, two_M)
            c12 == 0.0 && continue
            for k4 in 1:N
                two_m3 = two_M - (2k4 - 2 - two_j)
                k3 = (two_m3 + two_j) ÷ 2 + 1
                (1 <= k3 <= N && iseven(two_m3 + two_j)) || continue
                c43 = cg(two_j, 2k4 - 2 - two_j, two_j, two_m3, two_J, two_M)
                c43 == 0.0 && continue
                U[k1, k2, k3, k4] += V * c12 * c43
            end
        end
    end
    return U
end

# -----------------------------------------------------------------------------
# Channel weight projections (numerical, residual-asserted)
# -----------------------------------------------------------------------------

const _O2_CROSS = Dict{Any,Dict{Int,Float64}}()

"""c_lam in   2 sum_m U^{m1m2m3m4} (c+_f c_f)(c+_g c_g)
            = sum_lam c_lam sum_mu (-1)^mu T^lam_mu(f) T^lam_{-mu}(g)
for one unordered distinct flavor pair (f,g).  G[k1,k4,k2,k3] indexes the
bilinears (f: k1<-k4, g: k2<-k3)."""
function o2_cross_coefficients(N, ps)
    get!(_O2_CROSS, (N, Tuple(ps))) do
        U = o2_kernel(N, ps)
        G = zeros(N, N, N, N)
        for k1 in 1:N, k2 in 1:N, k3 in 1:N, k4 in 1:N
            G[k1, k4, k2, k3] += 2.0 * U[k1, k2, k3, k4]
        end
        out = Dict{Int,Float64}()
        reconstruction = zeros(size(G))
        for two_lam in 0:2:2*(N-1)
            basis = zeros(size(G))
            for two_mu in -two_lam:2:two_lam
                tf = t_coefficient_matrix(N, two_lam, two_mu)
                tg = t_coefficient_matrix(N, two_lam, -two_mu)
                p = phase(two_mu ÷ 2)
                @inbounds for k3 in 1:N, k2 in 1:N, k4 in 1:N, k1 in 1:N
                    basis[k1, k4, k2, k3] += p * tf[k1, k4] * tg[k2, k3]
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
        @assert residual < 1e-10 "o2 cross projection failed: $residual"
        out
    end
end

const _O2_EXCH = Dict{Any,Dict{Int,Float64}}()

"""c_lam in   [(0,a) exchange piece of Hxy]_NO  =  + sum U^{m1m2m3m4}
   B_a[m1,m3] B_0[m2,m4]   =  sum_lam c_lam sum_mu (-1)^mu T^lam_mu(a)
   T^lam_{-mu}(0),   projected on the distinguishable 2-particle space
(one active + one flavor-0 particle); residual asserted.  The same
coefficients serve both active flavors by the U index symmetry."""
function o2_exchange_coefficients(N, ps)
    get!(_O2_EXCH, (N, Tuple(ps))) do
        U = o2_kernel(N, ps)
        target = zeros(N * N, N * N)             # rows (k1,k2), cols (k3,k4)
        for k1 in 1:N, k2 in 1:N, k3 in 1:N, k4 in 1:N
            target[(k1-1)*N+k2, (k3-1)*N+k4] += U[k1, k2, k3, k4]
        end
        out = Dict{Int,Float64}()
        reconstruction = zeros(size(target))
        for two_lam in 0:2:2*(N-1)
            basis = zeros(size(target))
            for two_mu in -two_lam:2:two_lam
                ta = t_coefficient_matrix(N, two_lam, two_mu)
                t0 = t_coefficient_matrix(N, two_lam, -two_mu)
                p = phase(two_mu ÷ 2)
                @inbounds for k4 in 1:N, k3 in 1:N, k2 in 1:N, k1 in 1:N
                    basis[(k1-1)*N+k2, (k3-1)*N+k4] +=
                        p * ta[k1, k3] * t0[k2, k4]
                end
            end
            norm2 = sum(abs2, basis)
            norm2 < 1e-14 && continue
            c = dot(vec(basis), vec(target)) / norm2
            if abs(c) > 1e-14
                out[two_lam] = c
                reconstruction .+= c .* basis
            end
        end
        residual = maximum(abs, target .- reconstruction)
        @assert residual < 1e-10 "o2 exchange projection failed: $residual"
        out
    end
end

const _O2_TRANSFER = Dict{Any,Dict{Int,Float64}}()

"""d_K in   T_down = + sum U^{m1m2m3m4} [c+_{m1}]_+ [c+_{m2}]_- (c_{m4}c_{m3})_0
          = sum_K d_K sum_mu (-1)^mu [a+(+) x a+(-)]^K_mu Y^K_{-mu}(0)
(tensor-product tower form; the cross-tower fermion string (-1)^{n_+} is
applied at task level).  Projected in coefficient space, residual asserted."""
function o2_transfer_coefficients(N, ps)
    get!(_O2_TRANSFER, (N, Tuple(ps))) do
        two_j = N - 1
        U = o2_kernel(N, ps)
        # the symmetric-in-(m3,m4) part annihilates on the fermionic pair
        # string c_{m4} c_{m3}; only the antisymmetric part is the operator
        G = zeros(N, N, N, N)
        for k1 in 1:N, k2 in 1:N, k3 in 1:N, k4 in 1:N
            G[k1, k2, k3, k4] = 0.5 * (U[k1, k2, k3, k4] - U[k1, k2, k4, k3])
        end
        out = Dict{Int,Float64}()
        reconstruction = zeros(size(G))
        for two_K in 0:2:2*two_j
            basis = zeros(size(G))
            for two_mu in -two_K:2:two_K
                y = zeros(N, N)                  # string c_{ka} c_{kb}
                for (v, ops) in pair_y_terms(N, two_K, -two_mu)
                    y[ops[1][2]+1, ops[2][2]+1] += v
                end
                p = phase(two_mu ÷ 2)
                for k1 in 1:N
                    tm1 = 2 * (k1 - 1) - two_j
                    tm2 = two_mu - tm1
                    k2 = (tm2 + two_j) ÷ 2 + 1
                    (1 <= k2 <= N && iseven(tm2 + two_j)) || continue
                    c = cg(two_j, tm1, two_j, tm2, two_K, two_mu)
                    c == 0.0 && continue
                    @inbounds for k4 in 1:N, k3 in 1:N
                        basis[k1, k2, k3, k4] += p * c * y[k4, k3]
                    end
                end
            end
            norm2 = sum(abs2, basis)
            norm2 < 1e-14 && continue
            c = dot(vec(basis), vec(G)) / norm2
            if abs(c) > 1e-14
                out[two_K] = c
                reconstruction .+= c .* basis
            end
        end
        residual = maximum(abs, G .- reconstruction)
        @assert residual < 1e-10 "o2 transfer projection failed: $residual"
        out
    end
end

const _O2_SAME = Dict{Any,Dict{Int,Float64}}()

"""Two-particle matrix of a terms-list operator on one tower (states =
2-particle bitmasks over N orbitals)."""
function two_particle_matrix(N, terms)
    states = Int[]
    for k1 in 0:N-1, k2 in k1+1:N-1
        push!(states, (1 << k1) | (1 << k2))
    end
    index = Dict(s => i for (i, s) in enumerate(states))
    M = zeros(length(states), length(states))
    for (col, s) in enumerate(states), (value, ops) in terms
        out, sign = apply_ops(s, ops)
        sign == 0 && continue
        row = get(index, out, 0)
        row != 0 && (M[row, col] += sign * value)
    end
    return M
end

"""w_J in   [sum U (c+_f c_f)(c+_f c_f)]_one flavor  =  kappa n_f
          + sum_J w_J sum_M A+_{(ff)JM} A_{(ff)JM}.
Projected at the OPERATOR level on the 2-particle tower space (a two-body
operator is fully determined there), so no coefficient-tensor ordering
convention can silently break it; residual asserted."""
function o2_same_coefficients(N, ps)
    get!(_O2_SAME, (N, Tuple(ps))) do
        U = o2_kernel(N, ps)
        target_terms = Tuple{Float64,Vector{Tuple{Bool,Int}}}[]
        for k1 in 1:N, k2 in 1:N, k3 in 1:N, k4 in 1:N
            u = U[k1, k2, k3, k4]
            u == 0.0 && continue
            # NO part of the literal bilinear product (contraction excluded;
            # it is the kappa n_f piece, handled as an exact scalar)
            push!(target_terms, (u, [(true, k1 - 1), (true, k2 - 1),
                                     (false, k3 - 1), (false, k4 - 1)]))
        end
        G = two_particle_matrix(N, target_terms)
        out = Dict{Int,Float64}()
        reconstruction = zeros(size(G))
        for two_J in 0:2:2*(N-1)
            terms = Tuple{Float64,Vector{Tuple{Bool,Int}}}[]
            for two_M in -two_J:2:two_J
                create = pair_create_terms(N, two_J, two_M)
                for (v1, ops1) in create, (v2, ops2) in create
                    # A_{JM} = (A+_{JM})^dagger : reversed, daggers dropped
                    annih = [(false, ops2[2][2]), (false, ops2[1][2])]
                    push!(terms, (v1 * v2, vcat(ops1, annih)))
                end
            end
            B = two_particle_matrix(N, terms)
            norm2 = sum(abs2, B)
            norm2 < 1e-14 && continue
            c = dot(vec(B), vec(G)) / norm2
            if abs(c) > 1e-13
                out[two_J] = c
                reconstruction .+= c .* B
            end
        end
        residual = maximum(abs, G .- reconstruction)
        @assert residual < 1e-10 "o2 same-flavor projection failed: $residual"
        out
    end
end

# -----------------------------------------------------------------------------
# Sector
# -----------------------------------------------------------------------------

import SparseArrays

"""All (Jc, J0, a0) path-columns of one (n_+, n_-, J_+, J_-) pair are stored
contiguously: the frame's data is the (m_- m_+) x K matrix whose columns are
the paths.  Operators are two fat GEMMs on the row factors plus a column
transform stored as flat block entries: path-diagonal blocks (pm) or
(coef x shared r0 reference) blocks (cross, transfer).  Entries hold only
offsets, a scalar, and a reference, so memory stays ~30 bytes per coupled
path pair instead of an expanded sparse matrix."""
struct O2Frame
    n_plus::Int
    n_minus::Int
    two_Jp::Int
    two_Jm::Int
    mp::Int
    mm::Int
    offset::Int                                  # global start of frame data
    K::Int                                       # total path columns
    cols::Vector{NTuple{4,Int}}                  # (2Jc, 2J0, m0, colstart)
end

frame_K(f::O2Frame) = f.K

struct O2FrameTask
    fout::Int
    fin::Int
    rp::Union{Nothing,Matrix{Float64}}           # (m+', m+)
    rm::Union{Nothing,Matrix{Float64}}           # (m-', m-)
    # path-diagonal column blocks: (colstart in, colstart out, m0, coef)
    dpairs::Vector{Tuple{Int32,Int32,Int32,Float64}}
    # (colstart in, colstart out, coef, r0 ref (m0' x m0)) column blocks
    fentries::Vector{Tuple{Int32,Int32,Float64,Matrix{Float64}}}
end

mutable struct O2Sector
    N::Int
    D::Float64
    L::Int
    Q::Int
    dim::Int
    # channel: (n_+, n_-, 2J_0, 2J_+, 2J_-, 2Jc, m0, mp, mm)
    channels::Vector{NTuple{9,Int}}
    offset::Dict{NTuple{6,Int},Int}
    diagonal::Vector{Tuple{Int,Float64,Matrix{Float64},Matrix{Float64},
                           Matrix{Float64}}}     # off, scalar, w0, wp, wm
    diag_nd::Vector{Int}                         # d(scalar)/dD per entry
    frames::Vector{O2Frame}
    tasks::Vector{O2FrameTask}
end

"""Move the sector to anisotropy `D` in place: D enters the Hamiltonian only
through the diagonal scalars, so coupling scans re-use the whole build."""
function set_d!(sector::O2Sector, D::Float64)
    delta = D - sector.D
    delta == 0.0 && return sector
    for i in eachindex(sector.diagonal)
        off, sc, w0, wp, wm = sector.diagonal[i]
        sector.diagonal[i] = (off, sc + delta * sector.diag_nd[i],
                              w0, wp, wm)
    end
    sector.D = D
    return sector
end

function o2_channels(N, L, Q; n_pm_max = N)
    orb = orbit(N)
    two_L = 2L
    channels = NTuple{9,Int}[]
    offset = Dict{NTuple{6,Int},Int}()
    position = 0
    for n_plus in 0:N, n_minus in 0:N
        n_plus - n_minus == Q || continue
        n_plus + n_minus <= n_pm_max || continue
        n0 = N - n_plus - n_minus
        0 <= n0 <= N || continue
        plus = multiplets(orb, n_plus)
        minus = multiplets(orb, n_minus)
        zero = multiplets(orb, n0)
        for (two_Jp, mp) in sort(collect(plus); by = first),
            (two_Jm, mm) in sort(collect(minus); by = first)
            for (two_J0, m0) in sort(collect(zero); by = first)
                for two_Jc in abs(two_Jp - two_Jm):2:(two_Jp + two_Jm)
                    abs(two_Jc - two_J0) <= two_L <= two_Jc + two_J0 ||
                        continue
                    push!(channels, (n_plus, n_minus, two_J0, two_Jp,
                                     two_Jm, two_Jc, m0, mp, mm))
                    offset[(n_plus, n_minus, two_J0, two_Jp, two_Jm,
                            two_Jc)] = position
                    position += m0 * mp * mm
                end
            end
        end
    end
    return channels, offset, position
end

const _ADJ = Dict{Any,Matrix{Float64}}()

adjoint_of(key, M::Matrix{Float64}) = get!(() -> Matrix(M'), _ADJ, key)

"""Pieces: :pm, :p0, :m0 (H00 cross densities), :same, :xp0, :xm0 (Hxy
exchange), :xtr (Hxy pair transfer), :d.  `d_shift = false` keeps the literal
anisotropy D (paper convention already carries the exchange contraction
pieces explicitly, so the default full set uses no shift)."""
function O2Sector(N, D, ps_pot::Vector{Float64}, L, Q;
                  pieces = (:pm, :p0, :m0, :same, :xp0, :xm0, :xtr, :d),
                  d_shift = false, n_pm_max = N)
    orb = orbit(N)
    two_L = 2L
    channels, offset, dim = o2_channels(N, L, Q;
                                        n_pm_max = n_pm_max)
    c_lambda = o2_cross_coefficients(N, ps_pot)
    w_same = o2_same_coefficients(N, ps_pot)
    kappa = o2_kernel_kappa(N, ps_pot)
    diagonal = Tuple{Int,Float64,Matrix{Float64},Matrix{Float64},
                     Matrix{Float64}}[]
    diag_nd = Int[]

    for (n_plus, n_minus, two_J0, two_Jp, two_Jm, two_Jc, m0, mp, mm) in
        channels
        n0 = N - n_plus - n_minus
        scalar = 0.0
        if :d in pieces
            scalar += (D + (d_shift ? kappa / 2 : 0.0)) * (n_plus + n_minus)
        end
        if :same in pieces
            scalar += kappa * N                  # literal-form one-body piece
        end
        # exchange contraction one-body pieces (exact scalars)
        :xp0 in pieces && (scalar -= 0.5 * kappa * (n_plus + n0))
        :xm0 in pieces && (scalar -= 0.5 * kappa * (n_minus + n0))
        w0 = zeros(m0, m0)
        wp = zeros(mp, mp)
        wm = zeros(mm, mm)
        if :same in pieces
            for (two_J, w) in w_same
                w0 .+= w .* pair_scalar(orb, n0, two_J0, two_J)
                wp .+= w .* pair_scalar(orb, n_plus, two_Jp, two_J)
                wm .+= w .* pair_scalar(orb, n_minus, two_Jm, two_J)
            end
        end
        push!(diagonal, (offset[(n_plus, n_minus, two_J0, two_Jp, two_Jm,
                                 two_Jc)], scalar, w0, wp, wm))
        push!(diag_nd, :d in pieces ? n_plus + n_minus : 0)
    end

    # frames: maximal runs of consecutive channels sharing (n+, n-, J+, J-);
    # each frame's data is one contiguous (m_- m_+) x K matrix
    frames = O2Frame[]
    i = 1
    while i <= length(channels)
        c1 = channels[i]
        cols = NTuple{4,Int}[]
        colstart = 0
        j = i
        while j <= length(channels) &&
              channels[j][1] == c1[1] && channels[j][2] == c1[2] &&
              channels[j][4] == c1[4] && channels[j][5] == c1[5]
            ch = channels[j]
            push!(cols, (ch[6], ch[3], ch[7], colstart))
            colstart += ch[7]
            j += 1
        end
        push!(frames, O2Frame(c1[1], c1[2], c1[4], c1[5], c1[8], c1[9],
                              offset[c1[1:6]], colstart, cols))
        i = j
    end
    by_blockf = Dict{Tuple{Int,Int},Vector{Int}}()
    for (fi, f) in enumerate(frames)
        push!(get!(() -> Int[], by_blockf, (f.n_plus, f.n_minus)), fi)
    end
    cut = 1e-14

    tasks = O2FrameTask[]
    no_dp = Tuple{Int32,Int32,Int32,Float64}[]
    no_fe = Tuple{Int32,Int32,Float64,Matrix{Float64}}[]
    c_exch = (:xp0 in pieces || :xm0 in pieces) ?
             o2_exchange_coefficients(N, ps_pot) : Dict{Int,Float64}()
    d_K = :xtr in pieces ? o2_transfer_coefficients(N, ps_pot) :
          Dict{Int,Float64}()
    two_j = N - 1

    # ---- two-phase threaded task generation --------------------------------
    # Work items are (kind, fo, fi, two_lam): kind 1 = pm, 2/3 = cross
    # spectator :minus/:plus with c_lambda, 4/5 = same with c_exch,
    # 6 = transfer.  Phase 1 (serial) touches every mutable cache the full
    # pass will need (RMEs, adjoints); phase 2 runs the coefficient math
    # threaded -- its only mutable state is the per-thread 6j cache.

    # (+,-) density: both rme_t act; diagonal in the (Jc, J0) path
    function process_pm(fo, fi_, two_lam, c, warm, out)
        F_o, F_i = frames[fo], frames[fi_]
        rp = rme_t(orb, F_i.n_plus, two_lam, F_o.two_Jp, F_i.two_Jp)
        rm = rme_t(orb, F_i.n_minus, two_lam, F_o.two_Jm, F_i.two_Jm)
        (rp === nothing || rm === nothing || warm) && return
        dpairs = Tuple{Int32,Int32,Int32,Float64}[]
        for (Jc, J0, m0, cs_i) in F_i.cols
            coef = c * scalar_coef(F_o.two_Jp, F_o.two_Jm,
                                   F_i.two_Jp, F_i.two_Jm, Jc, two_lam)
            abs(coef) < cut && continue
            for (Jc2, J02, _, cs_o) in F_o.cols
                (Jc2 == Jc && J02 == J0) || continue
                push!(dpairs, (Int32(cs_i), Int32(cs_o), Int32(m0), coef))
            end
        end
        isempty(dpairs) && return
        push!(out, O2FrameTask(fo, fi_, rp, rm, dpairs, no_fe))
    end

    # (0,+) / (0,-) densities and exchange: one rme_t on the active tower,
    # column blocks (coef x r0 reference) with the folded chain scalar
    function process_cross(spectator, fo, fi_, two_lam, c, warm, out)
        F_o, F_i = frames[fo], frames[fi_]
        n0 = N - F_i.n_plus - F_i.n_minus
        if spectator == :minus
            r_act = rme_t(orb, F_i.n_plus, two_lam,
                          F_o.two_Jp, F_i.two_Jp)
            rp, rm = r_act, nothing
        else
            r_act = rme_t(orb, F_i.n_minus, two_lam,
                          F_o.two_Jm, F_i.two_Jm)
            rp, rm = nothing, r_act
        end
        if warm                                   # unique (J0, J02) pairs
            lastJ = (-1, -1)
            for (_, J0, _, _) in F_i.cols, (_, J02, _, _) in F_o.cols
                (J0, J02) == lastJ && continue
                lastJ = (J0, J02)
                abs(J02 - J0) <= two_lam &&
                    rme_t(orb, n0, two_lam, J02, J0)
            end
            return
        end
        r_act === nothing && return
        fentries = Tuple{Int32,Int32,Float64,Matrix{Float64}}[]
        for (Jc, J0, m0, cs_i) in F_i.cols,
            (Jc2, J02, m02, cs_o) in F_o.cols
            abs(J02 - J0) <= two_lam || continue
            r0 = rme_t(orb, n0, two_lam, J02, J0)
            r0 === nothing && continue
            coef = c * (spectator == :minus ?
                chain_scalar_p0(J02, F_o.two_Jp, F_o.two_Jm, Jc2, J0,
                                F_i.two_Jp, F_i.two_Jm, Jc, two_L,
                                two_lam) :
                chain_scalar_m0(J02, F_o.two_Jp, F_o.two_Jm, Jc2, J0,
                                F_i.two_Jp, F_i.two_Jm, Jc, two_L,
                                two_lam))
            abs(coef) * maximum(abs, r0) < cut && continue
            push!(fentries, (Int32(cs_i), Int32(cs_o), coef, r0))
        end
        isempty(fentries) && return
        push!(out, O2FrameTask(fo, fi_, rp, rm, no_dp, fentries))
    end

    # Hxy pair transfer: (n0 -> n0 - 2, n_+ + 1, n_- + 1) plus adjoint.
    # Coupled form: sum_K d_K [ [a+(+) x a+(-)]^K . Y^K(0) ] with the coupled
    # (+-) RME  sqrt((2Jc+1)(2Jc'+1)(2K+1)) 9j  (verified against explicit
    # CG) and the cross-tower fermion string (-1)^{n_+} of the (+,-,0) tower
    # order; one fentry per (path pair, K).
    function process_transfer(fo, fi_, warm, out)
        F_o, F_i = frames[fo], frames[fi_]
        n0 = N - F_i.n_plus - F_i.n_minus
        rp = rme_create1(orb, F_i.n_plus, F_o.two_Jp, F_i.two_Jp)
        rm = rme_create1(orb, F_i.n_minus, F_o.two_Jm, F_i.two_Jm)
        (rp === nothing || rm === nothing) && return
        if warm
            adjoint_of((:rc, N, F_i.n_plus, F_o.two_Jp, F_i.two_Jp), rp)
            adjoint_of((:rc, N, F_i.n_minus, F_o.two_Jm, F_i.two_Jm), rm)
            lastJ = (-1, -1)
            for (_, J0, _, _) in F_i.cols, (_, J02, _, _) in F_o.cols
                (J0, J02) == lastJ && continue
                lastJ = (J0, J02)
                for (two_K, _) in d_K
                    abs(J02 - J0) <= two_K || continue
                    r0 = rme_y(orb, n0, two_K, J02, J0)
                    r0 === nothing && continue
                    adjoint_of((:ty, N, n0, two_K, J02, J0), r0)
                end
            end
            return
        end
        fentries = Tuple{Int32,Int32,Float64,Matrix{Float64}}[]
        fadj = Tuple{Int32,Int32,Float64,Matrix{Float64}}[]
        for (Jc, J0, m0, cs_i) in F_i.cols,
            (Jc2, J02, m02, cs_o) in F_o.cols
            for (two_K, d) in d_K
                abs(Jc2 - Jc) <= two_K <= Jc2 + Jc || continue
                abs(J02 - J0) <= two_K || continue
                r0 = rme_y(orb, n0, two_K, J02, J0)
                r0 === nothing && continue
                nine = ninej(F_i.two_Jp, F_i.two_Jm, Jc, two_j, two_j,
                             two_K, F_o.two_Jp, F_o.two_Jm, Jc2)
                nine == 0.0 && continue
                coef = d * phase(F_i.n_plus) *
                       sqrt((Jc + 1.0) * (Jc2 + 1.0) * (two_K + 1.0)) *
                       nine * scalar_coef(Jc2, J02, Jc, J0, two_L, two_K)
                abs(coef) * maximum(abs, r0) < cut && continue
                push!(fentries, (Int32(cs_i), Int32(cs_o), coef, r0))
                push!(fadj, (Int32(cs_o), Int32(cs_i), coef,
                             adjoint_of((:ty, N, n0, two_K, J02, J0), r0)))
            end
        end
        isempty(fentries) && return
        push!(out, O2FrameTask(fo, fi_, rp, rm, no_dp, fentries))
        push!(out, O2FrameTask(
            fi_, fo,
            adjoint_of((:rc, N, F_i.n_plus, F_o.two_Jp, F_i.two_Jp), rp),
            adjoint_of((:rc, N, F_i.n_minus, F_o.two_Jm, F_i.two_Jm), rm),
            no_dp, fadj))
    end

    process(kind, fo, fi_, lam, warm, out) =
        kind == 1 ? process_pm(fo, fi_, lam, c_lambda[lam], warm, out) :
        kind == 2 ? process_cross(:minus, fo, fi_, lam, c_lambda[lam],
                                  warm, out) :
        kind == 3 ? process_cross(:plus, fo, fi_, lam, c_lambda[lam],
                                  warm, out) :
        kind == 4 ? process_cross(:minus, fo, fi_, lam, c_exch[lam],
                                  warm, out) :
        kind == 5 ? process_cross(:plus, fo, fi_, lam, c_exch[lam],
                                  warm, out) :
        process_transfer(fo, fi_, warm, out)

    # collect work items with cheap J-triangle guards only
    items = NTuple{4,Int}[]
    for (_, group) in by_blockf, fo in group, fi_ in group
        F_o, F_i = frames[fo], frames[fi_]
        dJp, dJm = abs(F_o.two_Jp - F_i.two_Jp), abs(F_o.two_Jm - F_i.two_Jm)
        if :pm in pieces
            for (two_lam, _) in c_lambda
                (dJp <= two_lam && dJm <= two_lam) &&
                    push!(items, (1, fo, fi_, two_lam))
            end
        end
        for (kind, cd, spec) in ((2, c_lambda, :minus), (3, c_lambda, :plus),
                                 (4, c_exch, :minus), (5, c_exch, :plus))
            piece = kind == 2 ? :p0 : kind == 3 ? :m0 :
                    kind == 4 ? :xp0 : :xm0
            piece in pieces || continue
            (spec == :minus ? dJm : dJp) == 0 || continue
            dact = spec == :minus ? dJp : dJm
            for (two_lam, _) in cd
                dact <= two_lam && push!(items, (kind, fo, fi_, two_lam))
            end
        end
    end
    if :xtr in pieces
        for fi_ in eachindex(frames)
            F_i = frames[fi_]
            N - F_i.n_plus - F_i.n_minus >= 2 || continue
            for fo in get(by_blockf, (F_i.n_plus + 1, F_i.n_minus + 1),
                          Int[])
                push!(items, (6, fo, fi_, 0))
            end
        end
    end

    # phase 1 (serial): warm every cache the threaded pass reads
    for (kind, fo, fi_, lam) in items
        process(kind, fo, fi_, lam, true, tasks)
    end
    # phase 2 (threaded): coefficient math; per-thread 6j caches, all other
    # caches now read-only
    nt = max(1, Threads.nthreads())
    lists = [O2FrameTask[] for _ in 1:nt]
    Threads.@threads :static for w in 1:nt
        for idx in w:nt:length(items)
            kind, fo, fi_, lam = items[idx]
            process(kind, fo, fi_, lam, false, lists[w])
        end
    end
    for l in lists
        append!(tasks, l)
    end
    merge_wigner_caches!()           # later builds hit the shared 6j and CG caches
    # restore frame-ordered task layout: the strided threading interleaves
    # tasks, which destroys matvec cache locality (measured 1.13 -> 1.8 s)
    sort!(tasks; by = t -> (t.fout, t.fin))

    return O2Sector(N, D, L, Q, dim, channels, offset, diagonal, diag_nd,
                    frames, tasks)
end

# -----------------------------------------------------------------------------
# Chain scalar coefficients for non-adjacent pairs, via explicit CG sums.
# Cached; small j's dominate at validation sizes, production uses the same
# routine (memoised) -- correctness first, closed 6j forms can replace these
# later behind the same verified interface.
# -----------------------------------------------------------------------------

"""<[(J+' J-)Jc', J0']L| sum_q (-1)^q T^k_q(+) T^k_{-q}(0) |[(J+ J-)Jc, J0]L>
 / (<J+'||T||J+> <J0'||T||J0>).  Closed form built ONLY from verified pieces:
the tensor on one pair member is [T^k x 1]^k with the coupled-RME 9j formula
(verified against explicit CG to 2e-15) and identity RME sqrt(2J+1), then the
adjacent-chain scalar_coef.  Not cached: the key space is per path pair (it
overwhelmed memory as a cache) while the 9j itself memoises its 6j parts.
`chain_scalar_generic` (explicit CG sums) remains as the reference;
o2_validate cross-checks the two on random samples."""
chain_scalar_p0(J0_o, Jp_o, Jm, Jc_o, J0_i, Jp_i, Jm_i, Jc_i, two_L, two_k) =
    sqrt((Jc_i + 1.0) * (Jc_o + 1.0) * (two_k + 1.0) * (Jm_i + 1.0)) *
    ninej(Jp_i, Jm_i, Jc_i, two_k, 0, two_k, Jp_o, Jm_i, Jc_o) *
    scalar_coef(Jc_o, J0_o, Jc_i, J0_i, two_L, two_k)

chain_scalar_m0(J0_o, Jp, Jm_o, Jc_o, J0_i, Jp_i, Jm_i, Jc_i, two_L, two_k) =
    sqrt((Jc_i + 1.0) * (Jc_o + 1.0) * (two_k + 1.0) * (Jp_i + 1.0)) *
    ninej(Jp_i, Jm_i, Jc_i, 0, two_k, two_k, Jp_i, Jm_o, Jc_o) *
    scalar_coef(Jc_o, J0_o, Jc_i, J0_i, two_L, two_k)

"""Explicit construction: couple (J+ J-)Jc then (Jc J0)L at M = L; apply
T^k(active) T^k(0) as CG-expanded single-tensor actions on the m-components;
divide by the Wigner-Eckart denominators of both active towers."""
function chain_scalar_generic(J0_o, Jp_o, Jm_o, Jc_o, J0_i, Jp_i, Jm_i, Jc_i,
                              two_L, two_k, active)
    amp(Jp, Jm, Jc, J0) = begin
        out = Dict{NTuple{3,Int},Float64}()
        for mp in -Jp:2:Jp, mm in -Jm:2:Jm
            mc = mp + mm
            abs(mc) > Jc && continue
            c1 = cg(Jp, mp, Jm, mm, Jc, mc)
            c1 == 0.0 && continue
            m0 = two_L - mc
            abs(m0) > J0 && continue
            c2 = cg(Jc, mc, J0, m0, two_L, two_L)
            c2 == 0.0 && continue
            out[(mp, mm, m0)] = get(out, (mp, mm, m0), 0.0) + c1 * c2
        end
        out
    end
    ket = amp(Jp_i, Jm_i, Jc_i, J0_i)
    bra = amp(Jp_o, Jm_o, Jc_o, J0_o)
    Ja_i, Ja_o = active == :plus ? (Jp_i, Jp_o) : (Jm_i, Jm_o)
    total = 0.0
    for two_q in -two_k:2:two_k, ((mp, mm, m0), a_k) in ket
        ma = active == :plus ? mp : mm
        t = cg(Ja_i, ma, two_k, two_q, Ja_o, ma + two_q)
        t == 0.0 && continue
        u = cg(J0_i, m0, two_k, -two_q, J0_o, m0 - two_q)
        u == 0.0 && continue
        key = active == :plus ? (mp + two_q, mm, m0 - two_q) :
                                (mp, mm + two_q, m0 - two_q)
        a_b = get(bra, key, 0.0)
        a_b == 0.0 && continue
        total += phase(two_q ÷ 2) * a_b * a_k * t * u
    end
    return total / sqrt((Ja_o + 1.0) * (J0_o + 1.0))
end

# -----------------------------------------------------------------------------
# Dense assembly (validation path)
# -----------------------------------------------------------------------------

function dense_o2(sector::O2Sector)
    matrix = zeros(sector.dim, sector.dim)
    eye(n) = Matrix{Float64}(I, n, n)
    for (off, scalar, w0, wp, wm) in sector.diagonal
        m0, mp, mm = size(w0, 1), size(wp, 1), size(wm, 1)
        sz = m0 * mp * mm
        block = kron(w0, eye(mp), eye(mm)) .+
                kron(eye(m0), wp, eye(mm)) .+
                kron(eye(m0), eye(mp), wm) .+ scalar .* eye(sz)
        matrix[off+1:off+sz, off+1:off+sz] .+= block
    end
    for t in sector.tasks
        F_o, F_i = sector.frames[t.fout], sector.frames[t.fin]
        rp = t.rp === nothing ? eye(F_o.mp) : t.rp
        rm = t.rm === nothing ? eye(F_o.mm) : t.rm
        rr = kron(rp, rm)                        # rows x column pair block
        so, si = size(rr)
        add!(col_o, col_i, v) =
            (matrix[F_o.offset+(col_o-1)*so+1:F_o.offset+col_o*so,
                    F_i.offset+(col_i-1)*si+1:F_i.offset+col_i*si] .+=
                 v .* rr)
        for (ci, co, m0w, c) in t.dpairs
            for a in 1:m0w
                add!(co + a, ci + a, c)
            end
        end
        for (ci, co, c, r0) in t.fentries
            m0o_, m0i_ = size(r0)
            for a0i in 1:m0i_, a0o in 1:m0o_
                v = c * r0[a0o, a0i]
                v == 0.0 && continue
                add!(co + a0o, ci + a0i, v)
            end
        end
    end
    return matrix
end

o2_eigenvalues(sector::O2Sector; k = 6) =
    sector.dim == 0 ? Float64[] :
    sort(eigvals(Symmetric(0.5 .* (dense_o2(sector) .+
                                   dense_o2(sector)'))))[1:min(k, sector.dim)]

# -----------------------------------------------------------------------------
# Iterative path: matvec over tasks (three-index contraction), Arpack solver
# -----------------------------------------------------------------------------

"""out[a, c, b] = in[a, b, c] for in of shape (A, B, C): the middle-to-last
permutation that turns the plus-contraction into one large GEMM."""
@inline function _permute23!(out, inp, A, B, C)
    @inbounds for c in 1:C, b in 1:B
        src = (c - 1) * A * B + (b - 1) * A
        dst = (b - 1) * A * C + (c - 1) * A
        @simd for a in 1:A
            out[dst+a] = inp[src+a]
        end
    end
end

"""y += (rp (x) rm) x (column transform)  on one frame pair; frame data is
the (m_- m_+) x (m0 nJc) path matrix.  `nothing` row factors are identities.
Middle-index contractions are reached by cheap permutations so every GEMM
has a fat dimension.  The column transform is path-diagonal blocks
(dpairs) or (coef x r0-reference) blocks (fentries).  `s1..s3` are
caller-owned scratch vectors (thread-private)."""
function _apply_frame_task!(y, x, t::O2FrameTask, F_o::O2Frame, F_i::O2Frame,
                            s1, s2, s3)
    mmi, mpi, Kin = F_i.mm, F_i.mp, frame_K(F_i)
    mmo, mpo, Kout = F_o.mm, F_o.mp, frame_K(F_o)
    X = reshape(view(x, F_i.offset+1:F_i.offset+mmi*mpi*Kin), mmi, mpi * Kin)
    if t.rm === nothing
        T1 = X                                   # (mmi = mmo, mpi * Kin)
    else
        T1 = reshape(view(s1, 1:mmo*mpi*Kin), mmo, mpi * Kin)
        mul!(T1, t.rm, X)
    end
    if t.rp === nothing
        T3 = reshape(T1, mmo * mpo, Kin)         # mpi == mpo
    else
        P = view(s2, 1:mmo*mpi*Kin)
        _permute23!(P, T1, mmo, mpi, Kin)
        Q = reshape(view(s1, 1:mmo*Kin*mpo), mmo * Kin, mpo)
        mul!(Q, reshape(P, mmo * Kin, mpi), transpose(t.rp))
        T3v = view(s3, 1:mmo*mpo*Kin)
        _permute23!(T3v, Q, mmo, Kin, mpo)
        T3 = reshape(T3v, mmo * mpo, Kin)
    end
    nr = mmo * mpo
    Y = reshape(view(y, F_o.offset+1:F_o.offset+nr*Kout), nr, Kout)
    @inbounds for (ci, co, m0w, c) in t.dpairs
        for a in 1:m0w
            col_i, col_o = ci + a, co + a
            @simd for r in 1:nr
                Y[r, col_o] += c * T3[r, col_i]
            end
        end
    end
    @inbounds for (ci, co, c, r0) in t.fentries
        m0o_, m0i_ = size(r0)
        for a0i in 1:m0i_
            col_i = ci + a0i
            for a0o in 1:m0o_
                v = c * r0[a0o, a0i]
                v == 0.0 && continue
                col_o = co + a0o
                @simd for r in 1:nr
                    Y[r, col_o] += v * T3[r, col_i]
                end
            end
        end
    end
    return nothing
end

struct O2Operator <: AbstractMatrix{Float64}
    sector::O2Sector
    ranges::Vector{UnitRange{Int}}               # flop-balanced task ranges
    buffers::Vector{Vector{Float64}}             # per-thread outputs
    s1s::Vector{Vector{Float64}}                 # per-thread scratch
    s2s::Vector{Vector{Float64}}
    s3s::Vector{Vector{Float64}}
end

function O2Operator(sector::O2Sector)
    frames = sector.frames
    flops = Float64[]
    max1 = max2 = max3 = 1
    for t in sector.tasks
        F_o, F_i = frames[t.fout], frames[t.fin]
        mmi, mpi, Kin = F_i.mm, F_i.mp, F_i.K
        mmo, mpo = F_o.mm, F_o.mp
        nr = mmo * mpo
        f = 0.0
        t.rm !== nothing && (f += mmo * mmi * mpi * Kin;
                             max1 = max(max1, mmo * mpi * Kin))
        if t.rp !== nothing
            f += mmo * mpi * mpo * Kin
            max1 = max(max1, mmo * Kin * mpo)
            max2 = max(max2, mmo * mpi * Kin)
            max3 = max(max3, mmo * mpo * Kin)
        end
        for (_, _, m0w, _) in t.dpairs
            f += nr * m0w
        end
        for (_, _, _, r0) in t.fentries
            f += nr * length(r0)
        end
        push!(flops, f)
    end
    nthreads = max(1, Threads.nthreads())
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
    return O2Operator(sector, ranges,
                      [zeros(sector.dim) for _ in 1:nthreads],
                      [zeros(max1) for _ in 1:nthreads],
                      [zeros(max2) for _ in 1:nthreads],
                      [zeros(max3) for _ in 1:nthreads])
end

Base.size(op::O2Operator) = (op.sector.dim, op.sector.dim)
Base.size(op::O2Operator, i::Int) = op.sector.dim
Base.eltype(::O2Operator) = Float64
LinearAlgebra.ishermitian(::O2Operator) = true

function LinearAlgebra.mul!(y::AbstractVector, op::O2Operator,
                            x::AbstractVector)
    sector = op.sector
    tasks = sector.tasks
    frames = sector.frames
    nthreads = length(op.buffers)
    Threads.@threads :static for w in 1:nthreads
        buffer = op.buffers[w]
        fill!(buffer, 0.0)
        s1, s2, s3 = op.s1s[w], op.s2s[w], op.s3s[w]
        for t in op.ranges[w]
            task = tasks[t]
            _apply_frame_task!(buffer, x, task, frames[task.fout],
                               frames[task.fin], s1, s2, s3)
        end
    end
    fill!(y, 0.0)
    for buffer in op.buffers
        y .+= buffer
    end
    # channel-diagonal part: disjoint y slices, threaded over channels
    Threads.@threads :static for i in eachindex(sector.diagonal)
        off, scalar, w0, wp, wm = sector.diagonal[i]
        m0, mp, mm = size(w0, 1), size(wp, 1), size(wm, 1)
        X = reshape(view(x, off+1:off+mm*mp*m0), mm, mp, m0)
        Y = reshape(view(y, off+1:off+mm*mp*m0), mm, mp, m0)
        Y .+= scalar .* X
        for a0 in 1:m0
            mul!(view(Y, :, :, a0), wm, view(X, :, :, a0), 1.0, 1.0)
            mul!(view(Y, :, :, a0), view(X, :, :, a0), wp, 1.0, 1.0)
        end
        mul!(reshape(Y, mm * mp, m0), reshape(X, mm * mp, m0),
             w0, 1.0, 1.0)                        # w symmetric
    end
    return y
end

"""Lowest k levels; dense below `dense_limit`, ARPACK above (warm-startable
via `v0`, as in the two-flavor solver)."""
function o2_solve(sector::O2Sector; k = 2, dense_limit = 3000, tol = 1e-9,
                  v0 = nothing)
    sector.dim == 0 && return Float64[]
    sector.dim <= dense_limit &&
        return o2_eigenvalues(sector; k = min(k, sector.dim))
    HAVE_ARPACK || error("Arpack.jl required for dim > $dense_limit")
    # small per-task GEMMs from many Julia threads: keep BLAS single-threaded
    # to avoid pool contention (measured on the two-flavor solver)
    Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)
    op = O2Operator(sector)
    kwargs = v0 === nothing ? (;) : (; v0 = v0 ./ norm(v0))
    values, _ = Arpack.eigs(op; nev = k, which = :SR, tol = tol,
                            ncv = min(sector.dim - 1, max(20, 10 * k)),
                            maxiter = 3000, kwargs...)
    return sort(real.(values))
end

# -----------------------------------------------------------------------------
# Validation: internal brute force + cross-code anchors
# -----------------------------------------------------------------------------

"""Literal m-scheme Hamiltonian on one (Lz, Q) block: bilinear products as
4-operator strings on 3N-bit states (spin-orbital i = 3k + f, flavors
f = 0,1,2 <-> sigma = +1,0,-1).  Independent of every J-scheme code path."""
function o2_brute_block(N, ps, D, two_lz, Q)
    two_j = N - 1
    w_q = (1, 0, -1)
    states = Int[]
    m = (1 << N) - 1                             # N particles set
    limit = 1 << (3N)
    while m < limit
        tlz, q = 0, 0
        bits = m
        while bits != 0
            i = trailing_zeros(bits)
            tlz += 2 * (i ÷ 3) - two_j
            q += w_q[i%3+1]
            bits &= bits - 1
        end
        (tlz == two_lz && q == Q) && push!(states, m)
        low = m & -m
        ripple = m + low
        m = ripple | (((m ⊻ ripple) >> 2) ÷ low)
    end
    index = Dict(s => i for (i, s) in enumerate(states))
    dim = length(states)
    U = o2_kernel(N, ps)
    # flavor-matrix entries (row, col) with piece weights
    density = [(0, 0), (1, 1), (2, 2)]
    prods = Tuple{Tuple{Int,Int},Tuple{Int,Int},Float64}[]
    for a in density, b in density
        push!(prods, (a, b, 1.0))                # H00, all ordered pairs
    end
    A, At, Z, Zt = (0, 1), (1, 0), (1, 2), (2, 1)
    for (a, b) in ((A, At), (At, A), (Z, Zt), (Zt, Z),
                   (A, Zt), (Zt, A), (Z, At), (At, Z))
        push!(prods, (a, b, -0.5))               # Hxy expansion
    end
    H = zeros(dim, dim)
    for k1 in 1:N, k2 in 1:N, k3 in 1:N, k4 in 1:N
        u = U[k1, k2, k3, k4]
        u == 0.0 && continue
        for ((ra, ca), (rb, cb), w) in prods
            ops = [(true, 3 * (k1 - 1) + ra), (false, 3 * (k4 - 1) + ca),
                   (true, 3 * (k2 - 1) + rb), (false, 3 * (k3 - 1) + cb)]
            for (col, s) in enumerate(states)
                out, sign = apply_ops(s, ops)
                sign == 0 && continue
                row = get(index, out, 0)
                row != 0 && (H[row, col] += sign * w * u)
            end
        end
    end
    for (col, s) in enumerate(states)
        nd = 0
        bits = s
        while bits != 0
            i = trailing_zeros(bits)
            i % 3 != 1 && (nd += 1)
            bits &= bits - 1
        end
        H[col, col] += D * nd
    end
    return sort(eigvals(Symmetric(0.5 .* (H .+ H'))))
end

"""Lowest k levels AND eigenvectors (dense below `dense_limit`, ARPACK
above)."""
function o2_eigensystem(sector::O2Sector; k = 2, dense_limit = 3000,
                        tol = 1e-9, v0 = nothing)
    sector.dim == 0 && return Float64[], zeros(0, 0)
    if sector.dim <= dense_limit
        M = dense_o2(sector)
        F = eigen(Symmetric(0.5 .* (M .+ M')))
        kk = min(k, sector.dim)
        return F.values[1:kk], F.vectors[:, 1:kk]
    end
    HAVE_ARPACK || error("Arpack.jl required for dim > $dense_limit")
    Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)
    op = O2Operator(sector)
    kwargs = v0 === nothing ? (;) : (; v0 = v0 ./ norm(v0))
    vals, vecs = Arpack.eigs(op; nev = k, which = :SR, tol = tol,
                             ncv = min(sector.dim - 1, max(20, 10 * k)),
                             maxiter = 3000, kwargs...)
    ix = sortperm(real.(vals))
    return real.(vals[ix]), real.(vecs[:, ix])
end

"""<v| n_+ + n_- |v>: the Hellmann-Feynman slope dE/dD (the anisotropy
operator is channel-diagonal).  Requires the sector built with the :d piece
(the default), which stores n_+ + n_- per diagonal entry in diag_nd."""
function o2_nd_expectation(sector::O2Sector, v::AbstractVector)
    total = 0.0
    for (i, entry) in enumerate(sector.diagonal)
        off, _, w0, wp, wm = entry
        len = size(w0, 1) * size(wp, 1) * size(wm, 1)
        s = 0.0
        @inbounds @simd for j in off+1:off+len
            s += v[j]^2
        end
        total += sector.diag_nd[i] * s
    end
    return total
end

"""Union of exact-L spectra over L >= Lz -- what an m-scheme (Lz, Q) block
must reproduce level by level."""
function o2_union(N, D, ps, Q, Lz; Lmax = N * (N - 1) ÷ 2 + 1)
    evs = Float64[]
    for L in Lz:Lmax
        s = O2Sector(N, D, ps, L, Q)
        s.dim == 0 && continue
        append!(evs, o2_eigenvalues(s; k = s.dim))
    end
    return sort(evs)
end

# FuzzifiED cross-code anchors: literal Hamiltonian (V0,V1) = (4,1) at
# D = 2.9600, lowest levels per charge sector (all L), measured in this
# session at D_FED = D + kappa/2 and reproduced by the oracle identity.
const O2_REFERENCE_N8 = Dict(
    0 => [23.68226059, 31.11270750, 33.36820456, 35.83548993, 37.22616221],
    1 => [26.32681489, 31.25501122, 36.27893808, 36.84220877, 38.14178328],
    2 => [30.20562072, 35.30573684, 38.97675003, 40.53513073, 41.36875006],
    3 => [35.15567492, 40.27024768, 44.43544423, 45.13154401, 46.43883687])

function o2_validate()
    passes = fails = 0
    check(name, ok, detail) = begin
        ok ? (passes += 1) : (fails += 1)
        @printf("  [%s] %s   %s\n", ok ? "PASS" : "FAIL", name, detail)
    end
    ps = [4.0, 1.0]

    worst = verify_scalar_formula()
    check("scalar-product 6j formula (re-fit)", worst < 1e-12,
          @sprintf("worst = %.1e", worst))

    # 6j canonicalization: every one of the 24 symmetry images must give the
    # same RAW value (computed without cache or canonical mapping)
    rngj = Random.MersenneTwister(17)
    worst = 0.0
    for _ in 1:200
        j1, j2, j4, j5 = rand(rngj, 0:9, 4)
        j3r = abs(j1 - j2):2:(j1 + j2)
        j6r = abs(j1 - j5):2:(j1 + j5)
        (isempty(j3r) || isempty(j6r)) && continue
        j3, j6 = rand(rngj, j3r), rand(rngj, j6r)
        ref = _sixj_compute(j1, j2, j3, j4, j5, j6)
        cols = ((j1, j4), (j2, j5), (j3, j6))
        for p in ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2),
                  (3, 2, 1)), flip in 0:3
            a, b, c = cols[p[1]], cols[p[2]], cols[p[3]]
            aa = flip == 2 || flip == 3 ? (a[2], a[1]) : a
            bb = flip == 1 || flip == 3 ? (b[2], b[1]) : b
            cc = flip == 1 || flip == 2 ? (c[2], c[1]) : c
            worst = max(worst, abs(_sixj_compute(aa[1], bb[1], cc[1],
                                                 aa[2], bb[2], cc[2]) - ref))
        end
    end
    check("6j 24-fold symmetry (canonical key safety)", worst < 1e-12,
          @sprintf("worst = %.1e over 200 keys", worst))

    # closed-form chain scalars vs explicit CG reference
    rng0 = Random.MersenneTwister(5)
    worst = 0.0
    tried = 0
    while tried < 30
        Jp, Jm, J0 = rand(rng0, 0:8), rand(rng0, 0:8), rand(rng0, 0:8)
        k = 2 * rand(rng0, 0:3)
        Jpp = abs(Jp + 2 * rand(rng0, -2:2))
        J0p = abs(J0 + 2 * rand(rng0, -2:2))
        (abs(Jp - Jpp) <= k <= Jp + Jpp && abs(J0 - J0p) <= k <= J0 + J0p) ||
            continue
        Jc = rand(rng0, abs(Jp - Jm):2:(Jp + Jm))
        Jcp_r = max(abs(Jpp - Jm), abs(Jc - k)):2:min(Jpp + Jm, Jc + k)
        isempty(Jcp_r) && continue
        Jcp = rand(rng0, Jcp_r)
        L_r = max(abs(Jc - J0), abs(Jcp - J0p)):2:min(Jc + J0, Jcp + J0p)
        isempty(L_r) && continue
        L = rand(rng0, L_r)
        tried += 1
        worst = max(worst,
                    abs(chain_scalar_p0(J0p, Jpp, Jm, Jcp, J0, Jp, Jm, Jc,
                                        L, k) -
                        chain_scalar_generic(J0p, Jpp, Jm, Jcp, J0, Jp, Jm,
                                             Jc, L, k, :plus)),
                    abs(chain_scalar_m0(J0p, Jm, Jpp, Jcp, J0, Jm, Jp, Jc,
                                        L, k) -
                        chain_scalar_generic(J0p, Jm, Jpp, Jcp, J0, Jm, Jp,
                                             Jc, L, k, :minus)))
    end
    check("chain-scalar closed form vs explicit CG", worst < 1e-12,
          @sprintf("worst = %.1e over %d samples", worst, tried))

    # brute force, literal Hamiltonian, N = 4 and 5: every (Lz, Q) block
    for (N, blocks) in ((4, [(0, 0), (0, 2), (1, 0), (2, 0), (1, 4)]),
                        (5, [(0, 0), (0, 2), (1, 0), (2, 0), (3, 0)]))
        worst = 0.0
        for (Q, tlz) in blocks
            brute = o2_brute_block(N, ps, 2.9600, tlz, Q)
            union = o2_union(N, 2.9600, ps, Q, tlz ÷ 2)
            if length(brute) != length(union)
                worst = Inf
                break
            end
            isempty(brute) && continue
            worst = max(worst, maximum(abs.(brute .- union)))
        end
        check(@sprintf("N = %d literal H vs internal brute force", N),
              worst < 1e-9, @sprintf("max |dE| = %.1e", worst))
    end

    # FuzzifiED anchors at N = 8 (charge-resolved, absolute energies)
    worst = 0.0
    for (Q, ref) in O2_REFERENCE_N8
        union = o2_union(8, 2.9600, ps, Q, 0; Lmax = 8)
        worst = max(worst, maximum(abs.(union[1:length(ref)] .- ref)))
    end
    check("N = 8 vs FuzzifiED (literal D, 20 levels, 4 charges)",
          worst < 5e-8, @sprintf("max |dE| = %.1e", worst))

    # n_pm_max truncation: variational from above, exact at full cut
    e_full = o2_eigenvalues(O2Sector(6, 2.9600, ps, 0, 0); k = 1)[1]
    e_cut = [o2_eigenvalues(O2Sector(6, 2.9600, ps, 0, 0;
                                     n_pm_max = c); k = 1)[1]
             for c in (2, 4, 6)]
    ok = e_cut[1] >= e_cut[2] - 1e-12 && e_cut[2] >= e_cut[3] - 1e-12 &&
         e_cut[3] >= e_full - 1e-12 &&
         abs(o2_eigenvalues(O2Sector(6, 2.9600, ps, 0, 0;
                                     n_pm_max = 6); k = 1)[1] - e_full) < 1e-9
    check("n_pm_max variational and monotone (N = 6)", ok,
          @sprintf("E0: %.6f >= %.6f >= %.6f >= %.6f", e_cut[1], e_cut[2],
                   e_cut[3], e_full))

    # Hellmann-Feynman slope equals the exact finite difference in D
    worst = 0.0
    for (L, Q) in ((0, 1), (0, 0))
        s = O2Sector(6, 2.9600, ps, L, Q)
        vals, vecs = o2_eigensystem(s; k = 1)
        hf = o2_nd_expectation(s, view(vecs, :, 1))
        h = 1e-5
        ep = o2_eigenvalues(set_d!(s, 2.9600 + h); k = 1)[1]
        em = o2_eigenvalues(set_d!(s, 2.9600 - h); k = 1)[1]
        worst = max(worst, abs((ep - em) / 2h - hf))
    end
    check("Hellmann-Feynman dE/dD vs finite difference", worst < 1e-6,
          @sprintf("worst = %.1e", worst))

    # set_d! reproduces a fresh build at the new anisotropy
    worst = 0.0
    for (L, Q) in ((0, 0), (0, 2))
        s1 = O2Sector(6, 2.0, ps, L, Q)
        set_d!(s1, 3.1)
        s2 = O2Sector(6, 3.1, ps, L, Q)
        worst = max(worst, maximum(abs.(o2_eigenvalues(s1; k = 5) .-
                                        o2_eigenvalues(s2; k = 5))))
    end
    check("set_d! equals fresh build", worst < 1e-10,
          @sprintf("max |dE| = %.1e", worst))

    # iterative matvec equals the dense assembly
    rng = Random.MersenneTwister(3)
    worst = 0.0
    for (L, Q) in ((0, 0), (1, 1), (2, 0), (0, 3))
        s = O2Sector(6, 2.9600, ps, L, Q)
        s.dim == 0 && continue
        M = dense_o2(s)
        x = randn(rng, s.dim)
        y = zeros(s.dim)
        mul!(y, O2Operator(s), x)
        worst = max(worst, maximum(abs.(M * x .- y)) / max(1.0, norm(x)))
    end
    check("matvec vs dense assembly (N = 6)", worst < 1e-10,
          @sprintf("worst = %.1e", worst))

    # charge conjugation: involution, commutation with H, and the parity
    # pattern that fixes the stress-tensor identification -- the lowest
    # (2,0) state is C-odd (the k=1 descendant of j), T is the SECOND,
    # C-even, state.  Getting this wrong is invisible in energies alone,
    # since both scaling dimensions converge to 3.
    let worst_inv = 0.0, worst_comm = 0.0, pattern_ok = true
        for (L, Q) in ((0, 0), (1, 0), (2, 0))
            s = O2Sector(8, 2.8747, ps, L, Q)
            perm, sgn = c_operator(s)
            v = randn(Random.MersenneTwister(41), s.dim)
            worst_inv = max(worst_inv,
                norm(c_apply(s, c_apply(s, v)) .- v) / norm(v))
            M = dense_o2(s)
            P = zeros(s.dim, s.dim)
            for i in 1:s.dim; P[perm[i], i] = sgn[i]; end
            worst_comm = max(worst_comm, opnorm(P * M .- M * P) / opnorm(M))
            _, vec = o2_eigensystem(s; k = 2)
            p1 = c_parity(s, vec[:, 1]); p2 = c_parity(s, vec[:, 2])
            if (L, Q) == (2, 0)
                pattern_ok &= p1 < -0.999 && p2 > 0.999
            elseif (L, Q) == (1, 0)
                pattern_ok &= p1 < -0.999
            else
                pattern_ok &= p1 > 0.999 && p2 > 0.999
            end
        end
        check("charge conjugation C: involution", worst_inv < 1e-12,
              @sprintf("worst = %.1e", worst_inv))
        check("charge conjugation C: [H, C] = 0", worst_comm < 1e-12,
              @sprintf("worst = %.1e", worst_comm))
        check("C parities: (2,0) lowest is C-odd d(j), T is 2nd (C-even)",
              pattern_ok, "pattern (0,0):++  (1,0):-  (2,0):-+")
    end

    # projected solve: the C-even ground state of (2,0) IS the stress tensor,
    # reachable with k = 1, so the k = 2 full-block solve is not needed.
    let worst = 0.0
        for N in (9, 10)
            s = O2Sector(N, N == 9 ? 2.8516 : 2.8347, ps, 2, 0)
            ev, _ = o2_eigensystem(s; k = 2)
            vm, _ = o2_ground_cparity(s, -1)
            vp, _ = o2_ground_cparity(s, +1)
            worst = max(worst, abs(vm - ev[1]), abs(vp - ev[2]))
        end
        check("C-projected k=1 reproduces the full k=2 pair", worst < 1e-10,
              @sprintf("worst = %.1e", worst))
    end

    @printf("\n%d/%d checks passed\n", passes, passes + fails)
    return fails == 0 ? 0 : 1
end

# ---------------------------------------------------------------------------
# Charge conjugation C : Q -> -Q
#
# The exact-(L,Q) blocks resolve L and Q but NOT C, which acts inside Q = 0 and
# splits that sector into the C-even and C-odd parts (the S = 0+ and S = 0- of
# Dey et al.).  Without it the lowest (L=2,Q=0) state is the C-odd k=1
# descendant of j_mu, not the stress tensor, and the two are easily confused
# because both scaling dimensions converge to 3.
#
# On the coupled basis C simply exchanges the two charged towers,
#   |[(J+ J-)Jc, J0] L>  ->  (-1)^{n+ n-} (-1)^{J+ + J- - Jc} |[(J- J+)Jc, J0] L>,
# the first factor from reordering the two blocks of fermions and the second
# from recoupling [(a b)c] -> [(b a)c].  Within a channel the storage order is
# kron(w0, I_mp, I_mm), so the multiplicity indices ip, im exchange too.
# The result is a signed permutation, and a symmetric involution.
"""Signed permutation representing C on a Q = 0 sector: `(perm, sgn)` with
`(C v)[perm[i]] = sgn[i] * v[i]`."""
function c_operator(S::O2Sector)
    S.Q == 0 || error("C is a symmetry only of the Q = 0 sector (got Q = $(S.Q))")
    perm = zeros(Int, S.dim); sgn = zeros(Float64, S.dim)
    for (n_plus, n_minus, two_J0, two_Jp, two_Jm, two_Jc, m0, mp, mm) in S.channels
        off  = S.offset[(n_plus, n_minus, two_J0, two_Jp, two_Jm, two_Jc)]
        off2 = S.offset[(n_minus, n_plus, two_J0, two_Jm, two_Jp, two_Jc)]
        ph = ((-1.0)^(n_plus * n_minus)) *
             ((-1.0)^(div(two_Jp + two_Jm - two_Jc, 2)))
        for i0 in 0:m0-1, ip in 0:mp-1, im in 0:mm-1
            src = off  + i0 * mp * mm + ip * mm + im
            dst = off2 + i0 * mm * mp + im * mp + ip
            perm[src+1] = dst + 1
            sgn[src+1]  = ph
        end
    end
    perm, sgn
end

"""Apply C to a coefficient vector."""
function c_apply(S::O2Sector, v::AbstractVector)
    perm, sgn = c_operator(S)
    out = similar(v)
    @inbounds for i in eachindex(v)
        out[perm[i]] = sgn[i] * v[i]
    end
    out
end

"""C parity of a state: +-1 for an eigenstate of H, since [H, C] = 0."""
c_parity(S::O2Sector, v::AbstractVector) = dot(v, c_apply(S, v)) / dot(v, v)

# ---------------------------------------------------------------------------
# C-projected eigensolve.
#
# [H, C] = 0 exactly, so projecting the Lanczos start vector onto a C-parity
# sector and re-projecting after every matvec confines the Krylov space to
# that sector; the sector's own ground state is then reached with k = 1.
# Re-projection each step is required because roundoff leaks toward the
# complementary sector, whose lowest state may lie BELOW the target (in
# (L=2, Q=0) the C-odd descendant of j sits below the C-even stress tensor).
struct O2ProjectedOperator <: AbstractMatrix{Float64}
    op::O2Operator
    perm::Vector{Int}
    sgn::Vector{Float64}
    parity::Float64
    tmp::Vector{Float64}
    xp::Vector{Float64}                          # projected input scratch
    mu::Float64                                  # complement shift
end
# The complementary C sector must be pushed UP, not annihilated: these blocks
# have positive energies, so a null space at 0 would sit below the physical
# spectrum and :SR would converge to it.  A = P H P + MU (1 - P).  MU only
# has to clear the top of the physical spectrum -- making it huge inflates
# the spectral range and destroys Lanczos convergence, so it is set from the
# block's own diagonal rather than fixed at some large constant.
Base.size(A::O2ProjectedOperator) = size(A.op)
Base.size(A::O2ProjectedOperator, i::Int) = size(A.op, i)
Base.eltype(::O2ProjectedOperator) = Float64
LinearAlgebra.issymmetric(::O2ProjectedOperator) = true
LinearAlgebra.ishermitian(::O2ProjectedOperator) = true

"""In place v <- (v + parity * C v)/2."""
function c_project!(v::AbstractVector, perm, sgn, parity, tmp)
    @inbounds for i in eachindex(v)
        tmp[perm[i]] = sgn[i] * v[i]
    end
    @inbounds for i in eachindex(v)
        v[i] = 0.5 * (v[i] + parity * tmp[i])
    end
    v
end

function LinearAlgebra.mul!(y::AbstractVector, A::O2ProjectedOperator,
                            x::AbstractVector)
    copyto!(A.xp, x)
    c_project!(A.xp, A.perm, A.sgn, A.parity, A.tmp)   # xp = P x
    mul!(y, A.op, A.xp)                                # y  = H P x
    c_project!(y, A.perm, A.sgn, A.parity, A.tmp)      # y  = P H P x
    @inbounds for i in eachindex(y)                    # y += MU (1-P) x
        y[i] += A.mu * (x[i] - A.xp[i])
    end
    y
end

"""Ground state of the given C-parity sector of a Q = 0 block: k = 1 Lanczos
with the Krylov space confined to that sector.  Returns (value, vector)."""
function o2_ground_cparity(sector::O2Sector, parity::Int;
                           tol = 1e-9, v0 = nothing)
    sector.Q == 0 || error("C parity is defined only at Q = 0")
    abs(parity) == 1 || error("parity must be +-1")
    perm, sgn = c_operator(sector)
    HAVE_ARPACK || error("Arpack.jl required")
    Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)
    # Gershgorin-free bound: the diagonal scalars already set the energy
    # scale of these blocks to within a factor of a few.
    dmax = 0.0
    for (_, sc, w0, wp, wm) in sector.diagonal
        dmax = max(dmax, abs(sc) + maximum(abs, w0) * size(w0, 1) +
                         maximum(abs, wp) * size(wp, 1) +
                         maximum(abs, wm) * size(wm, 1))
    end
    mu = 4.0 * dmax + 10.0
    A = O2ProjectedOperator(O2Operator(sector), perm, sgn, Float64(parity),
                            zeros(sector.dim), zeros(sector.dim), mu)
    start = v0 === nothing ? randn(sector.dim) : copy(v0)
    c_project!(start, perm, sgn, Float64(parity), A.tmp)
    n = norm(start)
    n < 1e-8 && (start = c_project!(randn(sector.dim), perm, sgn,
                                    Float64(parity), A.tmp); n = norm(start))
    vals, vecs = Arpack.eigs(A; nev = 1, which = :SR, tol = tol,
                             ncv = min(sector.dim - 1, 20),
                             maxiter = 3000, v0 = start ./ n)
    val = real(vals[1]); vec = real.(vecs[:, 1])
    # belt and braces: verify the vector really lies in the requested sector
    p = c_parity(sector, vec)
    abs(p - parity) < 1e-8 ||
        error("projected solve left the C sector: parity = $p")
    val, vec
end
