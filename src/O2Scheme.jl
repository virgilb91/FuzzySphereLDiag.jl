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
# The literal bilinear-product
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

"""One (J0 group, J0' group) block of a column transform.  Within a frame the
path columns run over J0 groups, then Jc, with the multiplicity index a0
fastest, and every path pair of the two groups carries the same reduced
matrix element r0 times a coupling coefficient that depends only on the two
Jc.  The block is therefore the Kronecker product S (x) r0, S of size
(nco, nci), applied as one GEMM over Jc and small ones over a0.  It replaces
one (coef, r0) entry per path pair: at N = 11, (L, Q) = (2, 0) those were
1.2e8 entries of ~7 multiply-adds each, 83% of the matvec.

S is stored column-major in its task's `svals` from offset `soff`, not as a
Matrix of its own: at N = 13, (L, Q) = (2, 0) there are 2e7 blocks, whose
Matrix headers and allocation slack took more memory than the coefficients."""
struct O2KronBlock
    ci::Int32                                    # colstart of the input J0 group
    co::Int32                                    # colstart of the output J0 group
    nci::Int32                                   # Jc paths in the input group
    nco::Int32                                   # Jc paths in the output group
    soff::Int32                                  # S = svals[soff .+ (1:nco*nci)]
    r0::Matrix{Float64}                          # (m0', m0), shared reference
end

"S[oo, ii] of a Kronecker block of task t."
@inline kron_s(t, b::O2KronBlock, oo, ii) = t.svals[b.soff+(ii-1)*b.nco+oo]

struct O2FrameTask
    fout::Int
    fin::Int
    rp::Union{Nothing,Matrix{Float64}}           # (m+', m+)
    rm::Union{Nothing,Matrix{Float64}}           # (m-', m-)
    # path-diagonal column blocks: (colstart in, colstart out, m0, coef)
    dpairs::Vector{Tuple{Int32,Int32,Int32,Float64}}
    # Kronecker column blocks S (x) r0 (cross, exchange and transfer pieces)
    fentries::Vector{O2KronBlock}
    svals::Vector{Float64}                       # the blocks' S coefficients
end

"""All (+-) density tasks of one frame pair, summed over the rank lambda.  Each
of them is path-diagonal, with a coefficient c_lam scalar_coef(J+', J-', J+, J-,
Jc, lam) that depends on the path only through Jc, so their sum acts on a path
column as one (mm' mp') x (mm mp) matrix
    M(Jc) = sum_lam c_lam(Jc) rp_lam (x) rm_lam,
applied once instead of once per lambda."""
struct O2PMTask
    fout::Int
    fin::Int
    mats::Vector{Matrix{Float64}}                # M(Jc), one per distinct Jc
    paths::Vector{NTuple{4,Int32}}               # (colstart in, colstart out, m0, mats index)
end


"""J0 groups of a frame: (2J0, m0, colstart, [2Jc...]) in storage order."""
function frame_groups(F::O2Frame)
    groups = Tuple{Int,Int,Int,Vector{Int}}[]
    for (Jc, J0, m0, cs) in F.cols
        if isempty(groups) || groups[end][1] != J0
            push!(groups, (J0, m0, cs, Int[]))
        end
        push!(groups[end][4], Jc)
    end
    return groups
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
    pmtasks::Vector{O2PMTask}  # merged (+-) density tasks, sorted by output frame
    cpar::Int                  # 0: full block; +-1: charge-conjugation parity (Q = 0)
    rep::Vector{Int}           # reduced basis vector k = (e_rep + coef e_partner)/sqrt2,
    partner::Vector{Int}       #   or e_rep alone where partner = 0
    coef::Vector{Float64}
end

"Dimension of the block actually diagonalised (the C half if cpar != 0)."
o2_rdim(s::O2Sector) = s.cpar == 0 ? s.dim : length(s.rep)

"""Reduced basis of one charge-conjugation half of a Q = 0 block.  C is the
signed permutation of `c_operator`: C e_i = sgn_i e_perm(i), exchanging the
two charged towers.  Kept frames are those with J_+ >= J_-: for J_+ > J_- every
entry i pairs with its image in the mirror frame, as (e_i + p sgn_i e_perm(i))/sqrt2;
a frame with J_+ = J_- is its own mirror, its entries with ip < im pair with
(im, ip), and its entries with ip = im are fixed by C, kept when sgn_i = p."""
function c_half_basis(channels, offset, parity)
    rep, partner, coef = Int[], Int[], Float64[]
    for (n_plus, n_minus, two_J0, two_Jp, two_Jm, two_Jc, m0, mp, mm) in channels
        two_Jp >= two_Jm || continue
        off  = offset[(n_plus, n_minus, two_J0, two_Jp, two_Jm, two_Jc)]
        off2 = offset[(n_minus, n_plus, two_J0, two_Jm, two_Jp, two_Jc)]
        sg = ((-1.0)^(n_plus * n_minus)) * ((-1.0)^(div(two_Jp + two_Jm - two_Jc, 2)))
        for i0 in 0:m0-1, ip in 0:mp-1, im in 0:mm-1
            src = off  + i0 * mp * mm + ip * mm + im + 1
            dst = off2 + i0 * mm * mp + im * mp + ip + 1
            if src == dst
                sg == parity && (push!(rep, src); push!(partner, 0); push!(coef, 0.0))
            elseif two_Jp > two_Jm || ip < im
                push!(rep, src); push!(partner, dst); push!(coef, parity * sg)
            end
        end
    end
    return rep, partner, coef
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
                  d_shift = false, n_pm_max = N, cpar = 0, merge_pm = true)
    cpar in (-1, 0, 1) || error("cpar must be -1, 0 or +1")
    cpar == 0 || Q == 0 || error("C parity is defined only at Q = 0 (got Q = $Q)")
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
    no_fe = O2KronBlock[]
    no_sv = Float64[]                            # shared empties: never mutated
    groups = [frame_groups(f) for f in frames]
    c_exch = (:xp0 in pieces || :xm0 in pieces) ?
             o2_exchange_coefficients(N, ps_pot) : Dict{Int,Float64}()
    # The H00 cross density and the Hxy exchange piece of one active flavor are
    # the same tensor operator sum_mu (-1)^mu T^lam_mu(a) T^lam_-mu(0) with
    # coefficients c_lambda and c_exch, so each (frame pair, lam) is one task
    # with the summed coefficient (half the tasks of building them apart).
    function merged(use_cross, use_exch)
        out = Dict{Int,Float64}()
        use_cross && mergewith!(+, out, c_lambda)
        use_exch && mergewith!(+, out, c_exch)
        return out
    end
    c_p0 = merged(:p0 in pieces, :xp0 in pieces)
    c_m0 = merged(:m0 in pieces, :xm0 in pieces)
    d_K = :xtr in pieces ? o2_transfer_coefficients(N, ps_pot) :
          Dict{Int,Float64}()
    two_j = N - 1

    # ---- two-phase threaded task generation --------------------------------
    # Work items are (kind, fo, fi, two_lam): kind 1 = pm, 2/3 = cross plus
    # exchange with spectator :minus/:plus (coefficients c_p0 / c_m0),
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
        push!(out, O2FrameTask(fo, fi_, rp, rm, dpairs, no_fe, no_sv))
    end

    # every rank lambda of the (+-) density on one frame pair at once: one
    # O2PMTask with M(Jc) = sum_lam c_lam(Jc) rp_lam (x) rm_lam when that costs
    # no more per column than the lambda tasks' row products and there are at
    # least two ranks; the separate process_pm tasks otherwise.  (Locals are
    # named apart from the constructor's: a closure shares any variable the
    # enclosing function also assigns.)
    function process_pm_frame(fo, fi_, warm, out, outpm)
        F_o, F_i = frames[fo], frames[fi_]
        dp_, dm_ = abs(F_o.two_Jp - F_i.two_Jp), abs(F_o.two_Jm - F_i.two_Jm)
        lamlist = Tuple{Int,Matrix{Float64},Matrix{Float64}}[]
        for (two_lam, _) in c_lambda
            (dp_ <= two_lam && dm_ <= two_lam) || continue
            rp_l = rme_t(orb, F_i.n_plus, two_lam, F_o.two_Jp, F_i.two_Jp)
            rm_l = rme_t(orb, F_i.n_minus, two_lam, F_o.two_Jm, F_i.two_Jm)
            (rp_l === nothing || rm_l === nothing) && continue
            push!(lamlist, (two_lam, rp_l, rm_l))
        end
        (warm || isempty(lamlist)) && return
        nr_o, nr_i = F_o.mm * F_o.mp, F_i.mm * F_i.mp
        fact = F_o.mm * F_i.mm * F_i.mp + F_o.mm * F_i.mp * F_o.mp
        if !(merge_pm && length(lamlist) >= 2 &&
             nr_o * nr_i <= length(lamlist) * min(nr_o * nr_i, fact))
            for (two_lam, _, _) in lamlist
                process_pm(fo, fi_, two_lam, c_lambda[two_lam], false, out)
            end
            return
        end
        krs = [kron(rp_l, rm_l) for (_, rp_l, rm_l) in lamlist]
        pm_index = Dict{Int,Int}()                   # Jc -> mats index, -1 if zero
        pm_mats = Matrix{Float64}[]
        pm_paths = NTuple{4,Int32}[]
        for (Jc_i, J0_i, m0_i, cs_i) in F_i.cols
            cs_o = -1
            for (Jc_o, J0_o, _, cso) in F_o.cols
                (Jc_o == Jc_i && J0_o == J0_i) && (cs_o = cso; break)
            end
            cs_o < 0 && continue
            idx = get(pm_index, Jc_i, 0)
            if idx == 0
                M = zeros(nr_o, nr_i)
                for (q, (two_lam, _, _)) in enumerate(lamlist)
                    cl = c_lambda[two_lam] * scalar_coef(F_o.two_Jp, F_o.two_Jm,
                                                         F_i.two_Jp, F_i.two_Jm, Jc_i, two_lam)
                    abs(cl) < cut && continue
                    M .+= cl .* krs[q]
                end
                if iszero(M)
                    idx = -1
                else
                    push!(pm_mats, M)
                    idx = length(pm_mats)
                end
                pm_index[Jc_i] = idx
            end
            idx < 0 && continue
            push!(pm_paths, (Int32(cs_i), Int32(cs_o), Int32(m0_i), Int32(idx)))
        end
        isempty(pm_paths) && return
        push!(outpm, O2PMTask(fo, fi_, pm_mats, pm_paths))
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
        blocks = O2KronBlock[]
        svals = Float64[]
        # frame part of the chain scalar, per (Jc', Jc), filled on first use
        lo_i = abs(F_i.two_Jp - F_i.two_Jm)
        lo_o = abs(F_o.two_Jp - F_o.two_Jm)
        A = fill(NaN, (F_o.two_Jp + F_o.two_Jm - lo_o) ÷ 2 + 1,
                 (F_i.two_Jp + F_i.two_Jm - lo_i) ÷ 2 + 1)
        for (J0, m0, cs_i, Jcs_i) in groups[fi_], (J02, m02, cs_o, Jcs_o) in groups[fo]
            abs(J02 - J0) <= two_lam || continue
            r0 = rme_t(orb, n0, two_lam, J02, J0)
            r0 === nothing && continue
            rmax = maximum(abs, r0)
            S = zeros(length(Jcs_o), length(Jcs_i))
            for (ii, Jc) in enumerate(Jcs_i), (oo, Jc2) in enumerate(Jcs_o)
                abs(Jc2 - Jc) <= two_lam <= Jc2 + Jc || continue
                ia, ja = (Jc2 - lo_o) ÷ 2 + 1, (Jc - lo_i) ÷ 2 + 1
                a = A[ia, ja]
                if isnan(a)
                    a = spectator == :minus ?
                        chain_frame_p0(F_o.two_Jp, F_o.two_Jm, Jc2,
                                       F_i.two_Jp, F_i.two_Jm, Jc, two_lam) :
                        chain_frame_m0(F_o.two_Jp, F_o.two_Jm, Jc2,
                                       F_i.two_Jp, F_i.two_Jm, Jc, two_lam)
                    A[ia, ja] = a
                end
                a == 0.0 && continue
                coef = c * a * scalar_coef(Jc2, J02, Jc, J0, two_L, two_lam)
                abs(coef) * rmax < cut && continue
                S[oo, ii] = coef
            end
            any(!iszero, S) || continue
            push!(blocks, O2KronBlock(Int32(cs_i), Int32(cs_o), Int32(length(Jcs_i)),
                                      Int32(length(Jcs_o)), Int32(length(svals)), r0))
            append!(svals, S)
        end
        isempty(blocks) && return
        # exact-size copies: growth slack of the appended vectors was ~2 GB
        # at N = 13, (L, Q) = (2, 0)
        push!(out, O2FrameTask(fo, fi_, rp, rm, no_dp, copy(blocks), copy(svals)))
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
        fentries = O2KronBlock[]
        fadj = O2KronBlock[]
        sv_f = Float64[]
        sv_a = Float64[]
        # the 9j of the coupled (+-) RME does not see J0: one table per K,
        # over (Jc', Jc), filled on first use
        lo_i = abs(F_i.two_Jp - F_i.two_Jm)
        lo_o = abs(F_o.two_Jp - F_o.two_Jm)
        nines = Dict(two_K => fill(NaN, (F_o.two_Jp + F_o.two_Jm - lo_o) ÷ 2 + 1,
                                   (F_i.two_Jp + F_i.two_Jm - lo_i) ÷ 2 + 1)
                     for (two_K, _) in d_K)
        for (J0, m0, cs_i, Jcs_i) in groups[fi_], (J02, m02, cs_o, Jcs_o) in groups[fo]
            for (two_K, d) in d_K
                abs(J02 - J0) <= two_K || continue
                r0 = rme_y(orb, n0, two_K, J02, J0)
                r0 === nothing && continue
                rmax = maximum(abs, r0)
                S = zeros(length(Jcs_o), length(Jcs_i))
                tab = nines[two_K]
                for (ii, Jc) in enumerate(Jcs_i), (oo, Jc2) in enumerate(Jcs_o)
                    abs(Jc2 - Jc) <= two_K <= Jc2 + Jc || continue
                    ia, ja = (Jc2 - lo_o) ÷ 2 + 1, (Jc - lo_i) ÷ 2 + 1
                    nine = tab[ia, ja]
                    if isnan(nine)
                        nine = ninej(F_i.two_Jp, F_i.two_Jm, Jc, two_j, two_j,
                                     two_K, F_o.two_Jp, F_o.two_Jm, Jc2)
                        tab[ia, ja] = nine
                    end
                    nine == 0.0 && continue
                    coef = d * phase(F_i.n_plus) *
                           sqrt((Jc + 1.0) * (Jc2 + 1.0) * (two_K + 1.0)) *
                           nine * scalar_coef(Jc2, J02, Jc, J0, two_L, two_K)
                    abs(coef) * rmax < cut && continue
                    S[oo, ii] = coef
                end
                any(!iszero, S) || continue
                push!(fentries, O2KronBlock(Int32(cs_i), Int32(cs_o), Int32(length(Jcs_i)),
                                            Int32(length(Jcs_o)), Int32(length(sv_f)), r0))
                append!(sv_f, S)
                push!(fadj, O2KronBlock(Int32(cs_o), Int32(cs_i), Int32(length(Jcs_o)),
                                        Int32(length(Jcs_i)), Int32(length(sv_a)),
                                        adjoint_of((:ty, N, n0, two_K, J02, J0), r0)))
                append!(sv_a, permutedims(S))
            end
        end
        isempty(fentries) && return
        push!(out, O2FrameTask(fo, fi_, rp, rm, no_dp, copy(fentries), copy(sv_f)))
        push!(out, O2FrameTask(
            fi_, fo,
            adjoint_of((:rc, N, F_i.n_plus, F_o.two_Jp, F_i.two_Jp), rp),
            adjoint_of((:rc, N, F_i.n_minus, F_o.two_Jm, F_i.two_Jm), rm),
            no_dp, copy(fadj), copy(sv_a)))
    end

    process(kind, fo, fi_, lam, warm, out, outpm) =
        kind == 1 ? process_pm_frame(fo, fi_, warm, out, outpm) :
        kind == 2 ? process_cross(:minus, fo, fi_, lam, c_p0[lam], warm, out) :
        kind == 3 ? process_cross(:plus, fo, fi_, lam, c_m0[lam], warm, out) :
        process_transfer(fo, fi_, warm, out)

    # collect work items with cheap J-triangle guards only
    items = NTuple{4,Int}[]
    # a C half needs only the rows of the frames with J_+ >= J_-
    keep(F) = cpar == 0 || F.two_Jp >= F.two_Jm
    for (_, group) in by_blockf, fo in group, fi_ in group
        F_o, F_i = frames[fo], frames[fi_]
        keep(F_o) || continue
        dJp, dJm = abs(F_o.two_Jp - F_i.two_Jp), abs(F_o.two_Jm - F_i.two_Jm)
        :pm in pieces &&                         # one item per frame pair
            any(dJp <= l && dJm <= l for l in keys(c_lambda)) &&
            push!(items, (1, fo, fi_, 0))
        for (kind, cd, spec) in ((2, c_p0, :minus), (3, c_m0, :plus))
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
                (keep(frames[fo]) || keep(F_i)) &&     # makes both directions
                    push!(items, (6, fo, fi_, 0))
            end
        end
    end

    # phase 1 (serial): warm every cache the threaded pass reads
    for (kind, fo, fi_, lam) in items
        process(kind, fo, fi_, lam, true, tasks, O2PMTask[])
    end
    # phase 2 (threaded): coefficient math; per-thread 6j caches, all other
    # caches now read-only
    nt = max(1, Threads.nthreads())
    lists = [O2FrameTask[] for _ in 1:nt]
    pmlists = [O2PMTask[] for _ in 1:nt]
    Threads.@threads :static for w in 1:nt
        for it in w:nt:length(items)
            kind, fo, fi_, lam = items[it]
            process(kind, fo, fi_, lam, false, lists[w], pmlists[w])
        end
    end
    for l in lists
        append!(tasks, l)
    end
    pmtasks = reduce(vcat, pmlists; init = O2PMTask[])
    sort!(pmtasks; by = t -> (t.fout, t.fin))
    cpar == 0 || filter!(t -> keep(frames[t.fout]), tasks)
    merge_sixj_caches!()             # later builds hit the shared 6j cache
    # restore frame-ordered task layout: the strided threading interleaves
    # tasks, which destroys matvec cache locality (measured 1.13 -> 1.8 s)
    sort!(tasks; by = t -> (t.fout, t.fin))

    CACHE_DIR[] === nothing || save_tables(N)
    # (names chosen not to collide with the closures' locals: a closure that
    # assigns a name the enclosing function also assigns shares that variable)
    half_basis = cpar == 0 ? (Int[], Int[], Float64[]) :
                 c_half_basis(channels, offset, cpar)
    return O2Sector(N, D, L, Q, dim, channels, offset, diagonal, diag_nd,
                    frames, tasks, pmtasks, cpar, half_basis...)
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
o2_validate cross-checks the two on random samples.

The chain scalar factors into a part fixed by the frame pair (the 9j, which
does not see J0) and the path-dependent scalar_coef; `chain_frame_p0/m0` is the
first part, so a task can tabulate it once per (Jc, Jc') instead of once per
path pair.  Its 9j has a zero entry and is a single 6j (`ninej_zero22/21`)."""
chain_scalar_p0(J0_o, Jp_o, Jm, Jc_o, J0_i, Jp_i, Jm_i, Jc_i, two_L, two_k) =
    chain_frame_p0(Jp_o, Jm, Jc_o, Jp_i, Jm_i, Jc_i, two_k) *
    scalar_coef(Jc_o, J0_o, Jc_i, J0_i, two_L, two_k)

chain_scalar_m0(J0_o, Jp, Jm_o, Jc_o, J0_i, Jp_i, Jm_i, Jc_i, two_L, two_k) =
    chain_frame_m0(Jp, Jm_o, Jc_o, Jp_i, Jm_i, Jc_i, two_k) *
    scalar_coef(Jc_o, J0_o, Jc_i, J0_i, two_L, two_k)

chain_frame_p0(Jp_o, Jm, Jc_o, Jp_i, Jm_i, Jc_i, two_k) =
    sqrt((Jc_i + 1.0) * (Jc_o + 1.0) * (two_k + 1.0) * (Jm_i + 1.0)) *
    ninej_zero22(Jp_i, Jm_i, Jc_i, two_k, two_k, Jp_o, Jm_i, Jc_o)

chain_frame_m0(Jp, Jm_o, Jc_o, Jp_i, Jm_i, Jc_i, two_k) =
    sqrt((Jc_i + 1.0) * (Jc_o + 1.0) * (two_k + 1.0) * (Jp_i + 1.0)) *
    ninej_zero21(Jp_i, Jm_i, Jc_i, two_k, two_k, Jp_i, Jm_o, Jc_o)

"""9j symbols with one zero entry, as a single 6j (doubled arguments):
    {a b c; d 0 f; g h i} = d_bh d_df (-1)^(b+c+d+g) {a c b; i g d} / sqrt((2b+1)(2d+1))
    {a b c; 0 e f; g h i} = d_ag d_ef (-1)^(a+b+f+i) {c b a; h i f} / sqrt((2a+1)(2f+1))
Both follow from the corner case {j1 j2 j3; j4 j5 j3; j7 j7 0} = (-1)^(j2+j3+j4+j7)
{j1 j2 j3; j5 j4 j7} / sqrt((2j3+1)(2j7+1)) after one row and one column
exchange, which together leave a 9j unchanged.  o2_validate compares both with
the general `ninej`."""
ninej_zero22(a, b, c, d, f, g, h, i) =
    (b == h && d == f) ?
    phase((b + c + d + g) ÷ 2) * sixj(a, c, b, i, g, d) / sqrt((b + 1.0) * (d + 1.0)) : 0.0

ninej_zero21(a, b, c, e, f, g, h, i) =
    (a == g && e == f) ?
    phase((a + b + f + i) ÷ 2) * sixj(c, b, a, h, i, f) / sqrt((a + 1.0) * (f + 1.0)) : 0.0

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
    sector.cpar == 0 || error("dense_o2 needs the full block; build it with cpar = 0")
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
        for b in t.fentries
            m0o_, m0i_ = size(b.r0)
            for ii in 1:b.nci, oo in 1:b.nco
                sc = kron_s(t, b, oo, ii)
                sc == 0.0 && continue
                for a0i in 1:m0i_, a0o in 1:m0o_
                    v = sc * b.r0[a0o, a0i]
                    v == 0.0 && continue
                    add!(b.co + (oo - 1) * m0o_ + a0o, b.ci + (ii - 1) * m0i_ + a0i, v)
                end
            end
        end
    end
    for t in sector.pmtasks
        F_o, F_i = sector.frames[t.fout], sector.frames[t.fin]
        so, si = F_o.mm * F_o.mp, F_i.mm * F_i.mp
        for (ci, co, m0w, k) in t.paths, a in 1:m0w
            matrix[F_o.offset+(co+a-1)*so+1:F_o.offset+(co+a)*so,
                   F_i.offset+(ci+a-1)*si+1:F_i.offset+(ci+a)*si] .+= t.mats[k]
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

"""out[ooff + (a, c, b)] = in[ioff + (a, b, c)] for in of shape (A, B, C): the
middle-to-last permutation that turns the plus-contraction into one GEMM."""
@inline function _permute23!(out, ooff, inp, ioff, A, B, C)
    @inbounds for c in 1:C, b in 1:B
        src = ioff + (c - 1) * A * B + (b - 1) * A
        dst = ooff + (b - 1) * A * C + (c - 1) * A
        @simd for a in 1:A
            out[dst+a] = inp[src+a]
        end
    end
end

"""Multiply-adds per path column of the factorized row product (rm, then rp)."""
_row_flops(t, F_o, F_i) =
    (t.rm === nothing ? 0 : F_o.mm * F_i.mm * F_i.mp) +
    (t.rp === nothing ? 0 : F_o.mm * F_i.mp * F_o.mp)

"""T3 = (rp (x) rm) X_in for one task, an (mm' mp') x K_in block at the start of
`s3`.  With `rr` (the explicit Kronecker product, used when the row factors are
small, which they are for nearly every task: mm mp is 2 at the median and 16 at
most at N = 11) it is one GEMM; otherwise rm, then rp through the middle-index
permutations.  Every product goes through the in-house kernel `_gemm!`.
OpenBLAS called from many Julia threads serializes on its buffer pool, which had
left the 8-thread matvec no faster than one thread (N = 11, (L, Q) = (0, 0):
0.119 s against 0.126 s)."""
function _row_factor!(s3, x, t::O2FrameTask, rr, F_o::O2Frame, F_i::O2Frame, s1, s2)
    mmi, mpi, Kin = F_i.mm, F_i.mp, F_i.K
    mmo, mpo = F_o.mm, F_o.mp
    nr, nri = mmo * mpo, mmi * mpi
    if rr !== nothing
        @inbounds for i in 1:nr*Kin
            s3[i] = 0.0
        end
        _gemm!(s3, 0, nr, Kin, rr, 0, nr, x, F_i.offset, 1, nri, nri, 1.0)
        return nothing
    end
    if t.rm === nothing                          # T1 = X (mmi == mmo)
        src, soff = x, F_i.offset
    else                                         # T1 = rm X, (mmo, mpi Kin)
        @inbounds for i in 1:mmo*mpi*Kin
            s1[i] = 0.0
        end
        _gemm!(s1, 0, mmo, mpi * Kin, t.rm, 0, mmo, x, F_i.offset, 1, mmi, mmi, 1.0)
        src, soff = s1, 0
    end
    if t.rp === nothing                          # T3 = T1 (mpi == mpo)
        copyto!(s3, 1, src, soff + 1, nr * Kin)
        return nothing
    end
    _permute23!(s2, 0, src, soff, mmo, mpi, Kin)  # (mmo, Kin, mpi)
    @inbounds for i in 1:mmo*Kin*mpo
        s1[i] = 0.0
    end
    _gemm!(s1, 0, mmo * Kin, mpo, s2, 0, mmo * Kin, t.rp, 0, mpo, 1, mpi, 1.0)
    _permute23!(s3, 0, s1, 0, mmo, Kin, mpo)      # (mmo, mpo, Kin)
    return nothing
end

"""y += (rp (x) rm) x (column transform) on one frame pair; frame data is the
(m_- m_+) x (m0 nJc) path matrix.  The column transform is path-diagonal blocks
(dpairs) or Kronecker blocks S (x) r0 (fentries).  `s1..s4` are the caller's
scratch vectors."""
function _apply_frame_task!(y, x, t::O2FrameTask, rr, F_o::O2Frame, F_i::O2Frame,
                            s1, s2, s3, s4)
    _row_factor!(s3, x, t, rr, F_o, F_i, s1, s2)
    nr = F_o.mm * F_o.mp
    yoff = F_o.offset
    @inbounds for (ci, co, m0w, c) in t.dpairs
        for a in 1:m0w
            yo = yoff + (co + a - 1) * nr
            xo = (ci + a - 1) * nr
            @simd for r in 1:nr
                y[yo+r] += c * s3[xo+r]
            end
        end
    end
    # Kronecker blocks: Y[:, (a0', jc')] += sum S[jc', jc] r0[a0', a0] T3[:, (a0, jc)]
    for b in t.fentries
        m0o_, m0i_ = size(b.r0)
        nci, nco = Int(b.nci), Int(b.nco)
        if m0o_ == 1 && m0i_ == 1                # one GEMM over Jc, r0 as a scalar
            _gemm!(y, yoff + b.co * nr, nr, nco, s3, b.ci * nr, nr,
                   t.svals, b.soff, nco, 1, nci, b.r0[1, 1])
        else                                     # Jc first, then a0 per output path
            w = nr * m0i_
            @inbounds for i in 1:w*nco
                s4[i] = 0.0
            end
            _gemm!(s4, 0, w, nco, s3, b.ci * nr, w, t.svals, b.soff, nco, 1, nci, 1.0)
            for jo in 1:nco
                _gemm!(y, yoff + (b.co + (jo - 1) * m0o_) * nr, nr, m0o_,
                       s4, (jo - 1) * w, nr, b.r0, 0, m0o_, 1, m0i_, 1.0)
            end
        end
    end
    return nothing
end

"""y = (channel-diagonal part) x on the channels `chans` of one frame: the
scalar, then W_- and W_+ on the charged multiplicity indices and W_0 on the
neutral one (all three symmetric), through `_gemm!` like the tasks."""
function _apply_frame_diagonal!(y, x, sector, chans)
    @inbounds for i in chans
        off, scalar, w0, wp, wm = sector.diagonal[i]
        m0, mp, mm = size(w0, 1), size(wp, 1), size(wm, 1)
        for j in off+1:off+mm*mp*m0
            y[j] = scalar * x[j]
        end
        _gemm!(y, off, mm, mp * m0, wm, 0, mm, x, off, 1, mm, mm, 1.0)
        for a0 in 1:m0
            o = off + (a0 - 1) * mm * mp
            _gemm!(y, o, mm, mp, x, o, mm, wp, 0, 1, mp, mp, 1.0)
        end
        _gemm!(y, off, mm * mp, m0, x, off, mm * mp, w0, 0, 1, m0, m0, 1.0)
    end
    return nothing
end

"""H x over output frames.  Each frame is a disjoint block of rows, so workers
take whole frames from a shared counter, largest first, and write y directly:
the diagonal channels of the frame first, then every task whose output is the
frame.  No per-thread copies of y, no reduction, and no static split whose
slowest part sets the pace (the flop-balanced static ranges this replaces were
3-4x out of balance, because tiny tasks cost far more than their flops)."""
struct O2Operator <: AbstractMatrix{Float64}
    sector::O2Sector
    rr::Vector{Union{Nothing,Matrix{Float64}}}  # explicit row factor per task
    ftasks::Vector{UnitRange{Int}}               # tasks by output frame
    fpm::Vector{UnitRange{Int}}                  # merged (+-) tasks by output frame
    fdiag::Vector{UnitRange{Int}}                # diagonal channels by frame
    order::Vector{Int}                           # frames, most work first
    s1s::Vector{Vector{Float64}}                 # per-worker scratch
    s2s::Vector{Vector{Float64}}
    s3s::Vector{Vector{Float64}}
    s4s::Vector{Vector{Float64}}
end

function O2Operator(sector::O2Sector;
                    frames_out = [f for (f, F) in enumerate(sector.frames)
                                  if sector.cpar == 0 || F.two_Jp >= F.two_Jm],
                    rows = :auto)
    rows in (:auto, :explicit, :factorized) || error("rows = :auto, :explicit or :factorized")
    frames, tasks = sector.frames, sector.tasks
    issorted(tasks; by = t -> t.fout) || error("tasks must be sorted by output frame")
    nf = length(frames)
    # diagonal channels per frame: frames are maximal runs of channels
    fdiag = Vector{UnitRange{Int}}(undef, nf)
    let f = 1, start = 1, chs = sector.channels
        for c in 2:length(chs)+1
            if c > length(chs) || chs[c][[1, 2, 4, 5]] != chs[c-1][[1, 2, 4, 5]]
                sector.diagonal[start][1] == frames[f].offset ||
                    error("frame/channel layout mismatch at frame $f")
                fdiag[f] = start:c-1
                f += 1
                start = c
            end
        end
        f == nf + 1 || error("frame count mismatch")
    end
    ftasks = fill(1:0, nf)
    let i = 1
        while i <= length(tasks)
            j = i
            while j < length(tasks) && tasks[j+1].fout == tasks[i].fout
                j += 1
            end
            ftasks[tasks[i].fout] = i:j
            i = j + 1
        end
    end
    fpm = fill(1:0, nf)
    let pms = sector.pmtasks, i = 1
        while i <= length(pms)
            j = i
            while j < length(pms) && pms[j+1].fout == pms[i].fout
                j += 1
            end
            fpm[pms[i].fout] = i:j
            i = j + 1
        end
    end
    rr = Vector{Union{Nothing,Matrix{Float64}}}(nothing, length(tasks))
    cache = Dict{NTuple{6,UInt},Matrix{Float64}}()
    work = zeros(nf)
    max1 = max2 = max3 = max4 = 1
    eye(n) = Matrix{Float64}(I, n, n)
    for (k, t) in enumerate(tasks)
        F_o, F_i = frames[t.fout], frames[t.fin]
        mmi, mpi, Kin = F_i.mm, F_i.mp, F_i.K
        mmo, mpo = F_o.mm, F_o.mp
        nr, nri = mmo * mpo, mmi * mpi
        fact = _row_flops(t, F_o, F_i)
        f = 0.0
        explicit = rows == :auto ? nr * nri <= max(64, 2 * fact) : rows == :explicit
        if explicit                                       # explicit Kronecker
            key = (t.rp === nothing ? UInt(0) : objectid(t.rp),
                   t.rm === nothing ? UInt(0) : objectid(t.rm),
                   UInt(mpo), UInt(mpi), UInt(mmo), UInt(mmi))
            rr[k] = get!(cache, key) do
                kron(t.rp === nothing ? eye(mpo) : t.rp,
                     t.rm === nothing ? eye(mmo) : t.rm)
            end
            f += nr * nri * Kin
        else
            f += fact * Kin
            max1 = max(max1, mmo * mpi * Kin, mmo * Kin * mpo)
            max2 = max(max2, mmo * mpi * Kin)
        end
        max3 = max(max3, nr * Kin)
        for (_, _, m0w, _) in t.dpairs
            f += nr * m0w
        end
        for b in t.fentries
            m0o_, m0i_ = size(b.r0)
            f += nr * m0i_ * b.nci * b.nco + (length(b.r0) > 1 ? nr * m0i_ * m0o_ * b.nco : 0)
            max4 = max(max4, nr * m0i_ * b.nco)
        end
        work[t.fout] += f + 200.0                         # + per-task overhead
    end
    for fi in 1:nf
        F = frames[fi]
        work[fi] += F.mm * F.mp * F.K * (F.mm + F.mp + 8)
    end
    for t in sector.pmtasks
        F_o, F_i = frames[t.fout], frames[t.fin]
        for (_, _, m0w, _) in t.paths
            work[t.fout] += F_o.mm * F_o.mp * F_i.mm * F_i.mp * m0w + 50.0
        end
    end
    order = [f for f in sortperm(work; rev = true) if f in frames_out]
    nw = max(1, Threads.nthreads())
    return O2Operator(sector, rr, ftasks, fpm, fdiag, order,
                      [zeros(max1) for _ in 1:nw], [zeros(max2) for _ in 1:nw],
                      [zeros(max3) for _ in 1:nw], [zeros(max4) for _ in 1:nw])
end

Base.size(op::O2Operator) = (op.sector.dim, op.sector.dim)
Base.size(op::O2Operator, i::Int) = op.sector.dim
Base.eltype(::O2Operator) = Float64
LinearAlgebra.ishermitian(::O2Operator) = true

function _apply_frame!(y, x, op::O2Operator, f, s1, s2, s3, s4)
    sector = op.sector
    frames, tasks = sector.frames, sector.tasks
    _apply_frame_diagonal!(y, x, sector, op.fdiag[f])
    for k in op.ftasks[f]
        t = tasks[k]
        _apply_frame_task!(y, x, t, op.rr[k], frames[t.fout], frames[t.fin],
                           s1, s2, s3, s4)
    end
    for k in op.fpm[f]
        t = sector.pmtasks[k]
        F_o, F_i = frames[t.fout], frames[t.fin]
        nr, nri = F_o.mm * F_o.mp, F_i.mm * F_i.mp
        for (ci, co, m0w, j) in t.paths                 # Y[:, co+a] += M(Jc) X[:, ci+a]
            _gemm!(y, F_o.offset + co * nr, nr, m0w, t.mats[j], 0, nr,
                   x, F_i.offset + ci * nri, 1, nri, nri, 1.0)
        end
    end
    return nothing
end

function LinearAlgebra.mul!(y::AbstractVector, op::O2Operator, x::AbstractVector)
    next = Threads.Atomic{Int}(1)
    order = op.order
    @sync for w in eachindex(op.s1s)
        Threads.@spawn begin
            s1, s2, s3, s4 = op.s1s[w], op.s2s[w], op.s3s[w], op.s4s[w]
            while true
                k = Threads.atomic_add!(next, 1)
                k > length(order) && break
                _apply_frame!(y, x, op, order[k], s1, s2, s3, s4)
            end
        end
    end
    return y
end

"""Diagonal of H in the coupled basis: the channel-diagonal part (scalar and the
diagonals of W_0, W_+, W_-) plus, for every task that maps a frame to itself,
coef rp[a,a] rm[b,b] on its path-diagonal columns and S[j,j] r0[a0,a0] rp rm on
the diagonal of its same-J0-group Kronecker blocks.  The Davidson
preconditioner; checked against the dense matrix by o2_validate."""
function o2_hdiag(sector::O2Sector)
    d = zeros(sector.dim)
    for (off, scalar, w0, wp, wm) in sector.diagonal
        m0, mp, mm = size(w0, 1), size(wp, 1), size(wm, 1)
        for a0 in 1:m0, ap in 1:mp, am in 1:mm
            d[off+((a0-1)*mp+(ap-1))*mm+am] = scalar + w0[a0, a0] + wp[ap, ap] +
                                              wm[am, am]
        end
    end
    for t in sector.tasks
        t.fout == t.fin || continue
        F = sector.frames[t.fout]
        nr = F.mm * F.mp
        rd = vec([(t.rp === nothing ? 1.0 : t.rp[ap, ap]) *
                  (t.rm === nothing ? 1.0 : t.rm[am, am])
                  for am in 1:F.mm, ap in 1:F.mp])       # index (ap-1) mm + am
        for (ci, co, m0w, c) in t.dpairs
            ci == co || continue
            for a in 1:m0w, r in 1:nr
                d[F.offset+(ci+a-1)*nr+r] += c * rd[r]
            end
        end
        for b in t.fentries
            b.ci == b.co || continue
            m0 = size(b.r0, 1)
            for jj in 1:b.nci, a0 in 1:m0
                v = kron_s(t, b, jj, jj) * b.r0[a0, a0]
                v == 0.0 && continue
                col = b.ci + (jj - 1) * m0 + a0
                for r in 1:nr
                    d[F.offset+(col-1)*nr+r] += v * rd[r]
                end
            end
        end
    end
    for t in sector.pmtasks
        t.fout == t.fin || continue
        F = sector.frames[t.fout]
        nr = F.mm * F.mp
        for (ci, co, m0w, k) in t.paths
            ci == co || continue
            M = t.mats[k]
            for a in 1:m0w, r in 1:nr
                d[F.offset+(ci+a-1)*nr+r] += M[r, r]
            end
        end
    end
    return d
end

# ---- charge-conjugation halves (O2Sector(...; cpar = +-1)) -------------------

"""Reduced vector -> full coupled-basis vector of definite C parity."""
function c_expand!(x::AbstractVector{Float64}, s::O2Sector, xr::AbstractVector{Float64})
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
function c_reduce!(xr::AbstractVector{Float64}, s::O2Sector, x::AbstractVector{Float64})
    r = inv(sqrt(2.0))
    @inbounds for k in eachindex(s.rep)
        p = s.partner[k]
        xr[k] = p == 0 ? x[s.rep[k]] : r * (x[s.rep[k]] + s.coef[k] * x[p])
    end
    return xr
end

"""H on a C half.  The input is expanded to a full vector of definite parity;
H x then has the same parity, so its reduced coordinates follow from the
representative entries alone, which lie in the frames with J_+ >= J_- -- the
only output frames the half's task list and operator compute."""
struct O2COperator <: AbstractMatrix{Float64}
    op::O2Operator
    x::Vector{Float64}
    y::Vector{Float64}
end

O2COperator(s::O2Sector) = O2COperator(O2Operator(s), zeros(s.dim), zeros(s.dim))
Base.size(A::O2COperator) = (o2_rdim(A.op.sector), o2_rdim(A.op.sector))
Base.size(A::O2COperator, i::Int) = o2_rdim(A.op.sector)
Base.eltype(::O2COperator) = Float64
LinearAlgebra.ishermitian(::O2COperator) = true

function LinearAlgebra.mul!(yr::AbstractVector, A::O2COperator, xr::AbstractVector)
    s = A.op.sector
    c_expand!(A.x, s, xr)
    mul!(A.y, A.op, A.x)
    q = sqrt(2.0)
    @inbounds for k in eachindex(s.rep)
        yr[k] = s.partner[k] == 0 ? A.y[s.rep[k]] : q * A.y[s.rep[k]]
    end
    return yr
end

"""Iterative eigensolve of a block or of a C half: Davidson with the diagonal of
H (default) or ARPACK.  Both stop at |r| <= tol |E|.  Returns (values, vectors
in the full coupled basis, matvec count)."""
function _o2_iterative(sector::O2Sector, k, tol, v0, solver)
    # BLAS on one thread: the small dense steps of both solvers are faster so
    Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)
    half = sector.cpar != 0
    op = half ? O2COperator(sector) : O2Operator(sector)
    n = size(op, 1)
    start = v0 === nothing ? nothing : half ? c_reduce!(zeros(n), sector, v0) : v0
    if solver == :davidson
        dg = o2_hdiag(sector)
        values, reduced, nmult = davidson(op, half ? dg[sector.rep] : dg;
                                          k = k, tol = tol, v0 = start)
    else
        solver == :arpack || error("solver = :davidson or :arpack")
        HAVE_ARPACK || error("Arpack.jl required for the :arpack solver")
        kwargs = start === nothing ? (;) : (; v0 = start ./ norm(start))
        vals, vecs, _, _, nmult = Arpack.eigs(op; nev = k, which = :SR, tol = tol,
                                              ncv = min(n - 1, max(20, 10 * k)),
                                              maxiter = 3000, kwargs...)
        ix = sortperm(real.(vals))
        values, reduced = real.(vals[ix]), real.(vecs[:, ix])
    end
    half || return values, reduced, nmult
    full = zeros(sector.dim, length(values))
    for c in axes(reduced, 2)
        c_expand!(view(full, :, c), sector, view(reduced, :, c))
    end
    return values, full, nmult
end

"""Dense eigensystem of a C half from its reduced matrix (small blocks)."""
function _o2_half_dense(sector::O2Sector, k)
    A = O2COperator(sector)
    n = size(A, 1)
    M = zeros(n, n)
    e = zeros(n)
    for c in 1:n
        e[c] = 1.0
        mul!(view(M, :, c), A, e)
        e[c] = 0.0
    end
    F = eigen(Symmetric(0.5 .* (M .+ M')))
    kk = min(k, n)
    full = zeros(sector.dim, kk)
    for c in 1:kk
        c_expand!(view(full, :, c), sector, view(F.vectors, :, c))
    end
    return F.values[1:kk], full
end

"""Lowest k levels; dense below `dense_limit`, else Davidson (`solver =
:davidson`, the default) or ARPACK (`:arpack`), warm-startable via `v0`."""
function o2_solve(sector::O2Sector; k = 2, dense_limit = 3000, tol = 1e-9,
                  v0 = nothing, solver = :davidson)
    o2_rdim(sector) == 0 && return Float64[]
    if o2_rdim(sector) <= dense_limit
        sector.cpar == 0 || return _o2_half_dense(sector, k)[1]
        return o2_eigenvalues(sector; k = min(k, sector.dim))
    end
    return _o2_iterative(sector, k, tol, v0, solver)[1]
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

"""Lowest k levels AND eigenvectors (dense below `dense_limit`, else Davidson
or ARPACK as in `o2_solve`)."""
function o2_eigensystem(sector::O2Sector; k = 2, dense_limit = 3000,
                        tol = 1e-9, v0 = nothing, solver = :davidson)
    o2_rdim(sector) == 0 && return Float64[], zeros(sector.dim, 0)
    o2_rdim(sector) <= dense_limit && sector.cpar != 0 &&
        return _o2_half_dense(sector, k)
    if sector.dim <= dense_limit
        M = dense_o2(sector)
        F = eigen(Symmetric(0.5 .* (M .+ M')))
        kk = min(k, sector.dim)
        return F.values[1:kk], F.vectors[:, 1:kk]
    end
    values, vectors, _ = _o2_iterative(sector, k, tol, v0, solver)
    return values, vectors
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

    # 9j with a zero entry as one 6j, against the general 9j sum
    rngz = Random.MersenneTwister(11)
    worst = 0.0
    for _ in 1:300
        tri(x, y) = abs(x - y):2:(x + y)
        a, b, d = rand(rngz, 0:12, 3)
        c, g = rand(rngz, tri(a, b)), rand(rngz, tri(a, d))
        ir = intersect(tri(c, d), tri(g, b))
        isempty(ir) || (i = rand(rngz, ir);
            worst = max(worst, abs(ninej(a, b, c, d, 0, d, g, b, i) -
                                   ninej_zero22(a, b, c, d, d, g, b, i))))
        h = rand(rngz, tri(b, d))
        ir = intersect(tri(c, d), tri(a, h))
        isempty(ir) || (i = rand(rngz, ir);
            worst = max(worst, abs(ninej(a, b, c, 0, d, d, a, h, i) -
                                   ninej_zero21(a, b, c, d, d, a, h, i))))
    end
    check("9j with a zero entry = one 6j", worst < 1e-12,
          @sprintf("worst = %.1e over 600 samples", worst))

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
        for rows in (:auto, :explicit, :factorized)
            mul!(y, O2Operator(s; rows = rows), x)
            worst = max(worst, maximum(abs.(M * x .- y)) / max(1.0, norm(x)))
        end
    end
    check("matvec vs dense, explicit and factorized rows (N = 6)", worst < 1e-10,
          @sprintf("worst = %.1e", worst))

    # merged (+-) tasks: the same operator as the separate lambda tasks
    let worst = 0.0, nmerged = 0, rngm = Random.MersenneTwister(7)
        for (N, L, Q) in ((7, 0, 0), (7, 2, 0), (8, 1, 1), (8, 2, 0))
            sm = O2Sector(N, 2.9600, ps, L, Q)
            su = O2Sector(N, 2.9600, ps, L, Q; merge_pm = false)
            nmerged += length(sm.pmtasks)
            x = randn(rngm, sm.dim)
            ym, yu = zeros(sm.dim), zeros(sm.dim)
            mul!(ym, O2Operator(sm), x)
            mul!(yu, O2Operator(su), x)
            worst = max(worst, maximum(abs.(ym .- yu)) / norm(x),
                        maximum(abs.(o2_hdiag(sm) .- o2_hdiag(su))))
        end
        check("merged (+-) tasks = separate lambda tasks (matvec, diagonal)",
              worst < 1e-12 && nmerged > 0,
              @sprintf("worst = %.1e, %d merged tasks", worst, nmerged))
    end

    # Davidson preconditioner: o2_hdiag is the exact diagonal of H
    worst = 0.0
    for (L, Q) in ((0, 0), (1, 1), (2, 0), (3, 0))
        s = O2Sector(6, 2.9600, ps, L, Q)
        s.dim == 0 && continue
        worst = max(worst, maximum(abs.(diag(dense_o2(s)) .- o2_hdiag(s))))
    end
    check("o2_hdiag = diagonal of the dense matrix (N = 6)", worst < 1e-12,
          @sprintf("worst = %.1e", worst))

    # Davidson and ARPACK agree with the dense spectrum
    let s = O2Sector(8, 2.9600, ps, 2, 0), worst = 0.0
        ref = o2_eigenvalues(s; k = 3)
        for solver in (:davidson, :arpack)
            e = o2_solve(s; k = 3, dense_limit = 0, tol = 1e-10, solver = solver)
            worst = max(worst, maximum(abs.(e .- ref)))
        end
        check("Davidson and ARPACK vs dense (N = 8, (L, Q) = (2, 0), 3 levels)",
              worst < 1e-9, @sprintf("max |dE| = %.1e", worst))
    end

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

    # charge-conjugation halves: together they hold the whole block, each
    # vector has the requested parity, and the iterative path on a half
    # reproduces the C-resolved levels of the full block
    let worst = 0.0, dims_ok = true, par_ok = true
        for (L, N) in ((0, 7), (1, 7), (2, 6), (3, 7))
            full = O2Sector(N, 2.9600, ps, L, 0)
            full.dim == 0 && continue
            halves = [O2Sector(N, 2.9600, ps, L, 0; cpar = p) for p in (1, -1)]
            dims_ok &= sum(o2_rdim, halves) == full.dim
            lev = Float64[]
            for (h, p) in zip(halves, (1, -1))
                o2_rdim(h) == 0 && continue
                e, v = _o2_half_dense(h, o2_rdim(h))
                append!(lev, e)
                par_ok &= all(abs(c_parity(h, view(v, :, c)) - p) < 1e-10
                              for c in axes(v, 2))
            end
            worst = max(worst, maximum(abs.(sort(lev) .- o2_eigenvalues(full; k = full.dim))))
        end
        check("C halves: dims add up, parities, union = full spectrum", dims_ok && par_ok &&
              worst < 1e-10, @sprintf("max |dE| = %.1e", worst))
    end
    let s = O2Sector(8, 2.8747, ps, 2, 0), worst = 0.0
        ev = o2_eigenvalues(s; k = 2)                # lowest C-odd, then T (C-even)
        for solver in (:davidson, :arpack)
            em = o2_solve(O2Sector(8, 2.8747, ps, 2, 0; cpar = -1); k = 1,
                          dense_limit = 0, tol = 1e-10, solver = solver)
            ep = o2_solve(O2Sector(8, 2.8747, ps, 2, 0; cpar = +1); k = 1,
                          dense_limit = 0, tol = 1e-10, solver = solver)
            worst = max(worst, abs(em[1] - ev[1]), abs(ep[1] - ev[2]))
        end
        check("C halves, iterative: (2,0) C-odd = level 1, C-even (T) = level 2",
              worst < 1e-9, @sprintf("max |dE| = %.1e", worst))
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
