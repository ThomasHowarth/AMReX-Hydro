/**
 * \file hydro_plic_owkes_desjardins.cpp
 * \addtogroup PLIC
 * @{
 */

#include <hydro_plic_owkes_desjardins.H>
#include <hydro_plic_owkes_desjardins_K.H>

#include <AMReX_MultiFabUtil.H>

using namespace amrex::literals;

namespace PLIC::OwkesDesjardins {

namespace {

void check_input (amrex::MultiFab const& F, amrex::Geometry const& geom, std::string const& reconstruction)
{
    CheckMesh(F, geom);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(F.nGrow() >= nGrowF(reconstruction), "PLIC: F needs PLIC::nGrowF() ghost cells");
}

void check_faces (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& faces, amrex::MultiFab const& F, int ncomp)
{
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            faces[dir].boxArray() == amrex::convert(F.boxArray(), amrex::IntVect::TheDimensionVector(dir)) &&
            faces[dir].DistributionMap() == F.DistributionMap() && faces[dir].nComp() >= ncomp,
            "PLIC: the fluxes must be on the faces of the grids of F");
    }
}

} // namespace

void
ComputeFluxes (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& liquid,
               amrex::MultiFab const& F, FaceConstPtrs const& umac,
               amrex::Geometry const& geom, amrex::Real dt,
               std::string const& reconstruction)
{
    BL_PROFILE("PLIC::OwkesDesjardins::ComputeFluxes()");
    check_input(F, geom, reconstruction);
    check_faces(liquid, F, 1);

    // Cells near the interface
    amrex::iMultiFab band(F.boxArray(), F.DistributionMap(), 1, 1);
    InterfaceBand(band, F, 1);

    // PLIC plane of the mixed cells, valid up to the first ghost layer
    amrex::MultiFab plane(F.boxArray(), F.DistributionMap(), 4, 1);
    FitPlanes(plane, F, 1, reconstruction);

    LevelFluxes(liquid, nullptr, band, F, plane, umac, geom, dt, nullptr, 0, 0.0_rt, 0.0_rt);

    // Faces between boxes compute the same value, up to round-off across a periodic boundary: keep
    // one, so that the volume is conserved exactly
    for (auto& fl : liquid) {
        fl.OverrideSync(geom.periodicity());
    }
}

void
ComputeMomentumFluxes (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& mass,
                       amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& momentum,
                       amrex::MultiFab const& F, amrex::MultiFab const& velocity, int vel_comp,
                       FaceConstPtrs const& umac, FaceConstPtrs const& face_state,
                       amrex::Geometry const& geom, amrex::Real dt,
                       amrex::Real rho_l, amrex::Real rho_g,
                       std::string const& reconstruction)
{
    BL_PROFILE("PLIC::OwkesDesjardins::ComputeMomentumFluxes()");
    check_input(F, geom, reconstruction);
    check_faces(mass, F, 1);
    check_faces(momentum, F, AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(velocity.nGrow() >= 1);

    const amrex::BoxArray& grids = F.boxArray();
    const amrex::DistributionMapping& dmap = F.DistributionMap();
    amrex::iMultiFab band(grids, dmap, 1, 1);
    InterfaceBand(band, F, 1);
    amrex::MultiFab plane(grids, dmap, 4, 1);
    FitPlanes(plane, F, 1, reconstruction);

    // The cuts of the flux volumes near the interface: the liquid, mass and momentum of each piece
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> liquid;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> cut;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const amrex::BoxArray fba = amrex::convert(grids, amrex::IntVect::TheDimensionVector(dir));
        liquid[dir].define(fba, dmap, 1, 0);
        cut[dir].define(fba, dmap, 1 + AMREX_SPACEDIM, 0);
    }
    LevelFluxes(liquid, &cut, band, F, plane, umac, geom, dt, &velocity, vel_comp, rho_l, rho_g);

    // Away from the interface the mass flux follows from the liquid flux, and the momentum flux
    // carries the face state
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
        const amrex::Real dx = geom.CellSize(dir);
        const amrex::Real dxinv = geom.InvCellSize(dir);
        const amrex::Real dtdx = dt * dxinv;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mass[dir]); mfi.isValid(); ++mfi) {
            const amrex::Box& fbx = mfi.validbox();
            auto const& mf = mass[dir].array(mfi);
            auto const& pf = momentum[dir].array(mfi);
            auto const& lf = liquid[dir].const_array(mfi);
            auto const& um = umac[dir]->const_array(mfi);
            auto const& face = face_state[dir]->const_array(mfi);
            auto const& bd = band.const_array(mfi);
            auto const& mfx = cut[dir].const_array(mfi);
            amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::ignore_unused(k);
                const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
                if (bd(iv) != 0 || bd(iv - e) != 0) {
                    mf(iv) = dx * mfx(iv, 0);
                    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                        pf(iv, n) = dx * mfx(iv, 1 + n);
                    }
                } else {
                    const amrex::Real courant = um(iv) * dtdx;
                    const amrex::Real m = rho_g * courant + (rho_l - rho_g) * lf(iv) * dxinv;
                    mf(iv) = dx * m;
                    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                        pf(iv, n) = dx * m * face(iv, n);
                    }
                }
            });
        }
        amrex::Gpu::streamSynchronize();
        mass[dir].OverrideSync(geom.periodicity());
        momentum[dir].OverrideSync(geom.periodicity());
    }
}

void
LevelFluxes (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& liquid,
             amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>* mass_momentum,
             amrex::iMultiFab const& band,
             amrex::MultiFab const& F, amrex::MultiFab const& plane,
             FaceConstPtrs const& a_umac, amrex::Geometry const& geom, amrex::Real dt,
             amrex::MultiFab const* velocity, int vel_comp,
             amrex::Real rho_l, amrex::Real rho_g)
{
    // The advection is explicit. The step is usually estimated from the cell-centered velocity at
    // t^n, and the face velocity at t^{n+1/2} can be larger.
    amrex::Real umac_rate = 0.0_rt;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_umac[dir]->nGrow() >= 1,
                                         "PLIC: umac needs a filled ghost layer, for a node to be traced the same way in all the boxes it touches");
        umac_rate = amrex::max(umac_rate, a_umac[dir]->norm0(0, 0, true) * geom.InvCellSize(dir));
    }
    amrex::ParallelDescriptor::ReduceRealMax(umac_rate);
    if (umac_rate * dt >= 1.0_rt) {
        amrex::Abort("PLIC: the Owkes-Desjardins advection requires a face Courant number below 1: reduce the time step.");
    }

    const amrex::BoxArray& grids = F.boxArray();
    const amrex::DistributionMapping& dmap = F.DistributionMap();
    const amrex::Box& domain = geom.Domain();
    const auto periodicity = geom.periodicity();
    const amrex::IntVect is_periodic(AMREX_D_DECL(static_cast<int>(geom.isPeriodic(0)),
                                                  static_cast<int>(geom.isPeriodic(1)),
                                                  static_cast<int>(geom.isPeriodic(2))));
    const amrex::BoxArray nodal_ba = amrex::convert(grids, amrex::IntVect::TheNodeVector());

    // Nodes that can be traced to. Positions beyond the domain use its nearest nodes, except across
    // periodic boundaries.
    amrex::Box node_limit = amrex::surroundingNodes(domain);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (is_periodic[d] != 0) {
            node_limit.grow(d, 1000000);
        }
    }

    // Node velocities in Courant numbers: each component is the average of the faces normal to it
    // that touch the node, one-sided at domain boundaries. A node shared by several boxes gets one
    // value, so that it is traced to one position and the flux volumes tile across boxes. The ghost
    // nodes at coarse/fine boundaries use the ghost faces that are available.
    amrex::MultiFab nvel(nodal_ba, dmap, AMREX_SPACEDIM, 1);
    nvel.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(nvel); mfi.isValid(); ++mfi) {
        const amrex::Box nbx = amrex::grow(mfi.validbox(), 1) & node_limit;
        auto const& vel = nvel.array(mfi);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            auto const& um = a_umac[dir]->const_array(mfi);
            const amrex::Real dtdx = dt * geom.InvCellSize(dir);
            amrex::ParallelFor(nbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::ignore_unused(k);
                const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
                amrex::Real sum = 0.0_rt;
                int count = 0;
                const int t1 = (dir + 1) % AMREX_SPACEDIM;
                constexpr int n1 = 2;
                const int t2 = (AMREX_SPACEDIM == 3) ? (dir + 2) % AMREX_SPACEDIM : t1;
                const int n2 = (AMREX_SPACEDIM == 3) ? 2 : 1;
                for (int s2 = 0; s2 < n2; ++s2) {
                    for (int s1 = 0; s1 < n1; ++s1) {
                        amrex::IntVect f = iv;
                        f[t1] -= s1;
                        if (AMREX_SPACEDIM == 3) {
                            f[t2] -= s2;
                        }
                        bool inside = (is_periodic[t1] != 0) ||
                                      (f[t1] >= domain.smallEnd(t1) && f[t1] <= domain.bigEnd(t1));
                        inside = inside && ((AMREX_SPACEDIM != 3) || (is_periodic[t2] != 0) ||
                                            (f[t2] >= domain.smallEnd(t2) && f[t2] <= domain.bigEnd(t2)));
                        if (inside && um.contains(f)) {
                            sum += um(f);
                            ++count;
                        }
                    }
                }
                vel(i, j, k, dir) = sum * dtdx / static_cast<amrex::Real>(amrex::max(count, 1));
            });
        }
    }
    nvel.FillBoundary(periodicity);

    // Traced position of each node at the start of the step (midpoint rule), in cell units
    amrex::MultiFab back(nodal_ba, dmap, 3, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(back); mfi.isValid(); ++mfi) {
        const amrex::Box& nbx = mfi.validbox();
        const amrex::Box lim = amrex::grow(nbx, 1) & node_limit;
        auto const& vel = nvel.const_array(mfi);
        auto const& bk = back.array(mfi);
        amrex::ParallelFor(nbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const amrex::Real x[3] = {static_cast<amrex::Real>(i), static_cast<amrex::Real>(j),
                                      static_cast<amrex::Real>(k)};
            amrex::Real xm[3] = {x[0], x[1], x[2]};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                xm[d] -= 0.5_rt * vel(i, j, k, d);
            }
            amrex::Real um[3] = {0.0_rt, 0.0_rt, 0.0_rt};
            interp_node_velocity(xm, vel, lim, um);
            for (int d = 0; d < 3; ++d) {
                bk(i, j, k, d) = x[d] - um[d];
            }
        });
    }

    // Liquid crossing each face per unit face area. Near the interface it comes from the cuts of the
    // flux volume, with the mass and momentum. Away from it the flux volume, which stays in the cells
    // around the face for a Courant number below 1, is all in one phase.
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto& flux = liquid[dir];
        const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
        const amrex::Real dx = geom.CellSize(dir);
        const amrex::Real dtdx = dt / dx;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(flux); mfi.isValid(); ++mfi) {
            const amrex::Box& fbx = mfi.validbox();
            const amrex::Box cells = amrex::grow(amrex::enclosedCells(fbx), 1);
            auto const& fl = flux.array(mfi);
            auto const& um = a_umac[dir]->const_array(mfi);
            auto const& Fa = F.const_array(mfi);
            auto const& pl = plane.const_array(mfi);
            auto const& bk = back.const_array(mfi);
            auto const& bd = band.const_array(mfi);
            if (mass_momentum != nullptr) {
                auto const& mfx = (*mass_momentum)[dir].array(mfi);
                const amrex::Array4<amrex::Real const> vl(velocity->const_array(mfi), vel_comp);
                amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
                    const amrex::Real courant = um(i, j, k) * dtdx;
                    if (bd(iv) != 0 || bd(iv - e) != 0) {
                        amrex::Real m = 0.0_rt;
                        amrex::Real p[AMREX_SPACEDIM];
                        amrex::Real l = 0.0_rt;
                        face_mass_momentum_flux(i, j, k, dir, courant, Fa, pl, bk, vl, cells, rho_l, rho_g, m, p, l);
                        fl(i, j, k) = dx * l;
                        mfx(i, j, k, 0) = m;
                        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                            mfx(i, j, k, 1 + n) = p[n];
                        }
                    } else {
                        fl(i, j, k) = dx * ((Fa(i, j, k) > 0.5_rt) ? courant : 0.0_rt);
                    }
                });
            } else {
                amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
                    const amrex::Real courant = um(i, j, k) * dtdx;
                    if (bd(iv) != 0 || bd(iv - e) != 0) {
                        fl(i, j, k) = dx * face_liquid_flux(i, j, k, dir, courant, Fa, pl, bk, cells);
                    } else {
                        fl(i, j, k) = dx * ((Fa(i, j, k) > 0.5_rt) ? courant : 0.0_rt);
                    }
                });
            }
        }
    }
    amrex::Gpu::streamSynchronize();
}

void
InterfaceBand (amrex::iMultiFab& band, amrex::MultiFab const& F, int ngrow)
{
    AMREX_ALWAYS_ASSERT(band.nGrow() >= ngrow && F.nGrow() >= ngrow + 1);
    auto const& band_ma = band.arrays();
    auto const& F_ma = F.const_arrays();
    amrex::ParallelFor(band, amrex::IntVect(ngrow),
    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        band_ma[box_no](i, j, k) = static_cast<int>(interface_band(i, j, k, F_ma[box_no]));
    });
    amrex::Gpu::streamSynchronize();
}

} // namespace PLIC::OwkesDesjardins
/** @}*/
