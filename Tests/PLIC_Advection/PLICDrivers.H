#ifndef PLIC_DRIVERS_H_
#define PLIC_DRIVERS_H_

// How a solver advances the volume fraction with the operators of the PLIC library: here all levels
// with the same time step. A subcycling solver advances each level on its own instead, and refluxes
// with the same fluxes.

#include <hydro_plic.H>
#include <hydro_plic_owkes_desjardins.H>

#include <AMReX_MultiFabUtil.H>

#include <functional>
#include <string>

namespace drivers {

//! Fills the ghost cells of F on one level, from the valid cells of this level and the coarser ones
using FillGhostCells = std::function<void(int lev, amrex::MultiFab& F)>;

//! Face-centered data on the grids of F
inline amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
make_faces (amrex::MultiFab const& F, int ncomp)
{
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> faces;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        faces[dir].define(amrex::convert(F.boxArray(), amrex::IntVect::TheDimensionVector(dir)),
                          F.DistributionMap(), ncomp, 0);
    }
    return faces;
}

// ------------------------------------------------------------------------------- owkes_desjardins

//! F += sum_dir (liquid_lo - liquid_hi) / dx on the valid cells
inline void
UpdateFromFluxes (amrex::MultiFab& F, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& liquid,
                  amrex::Geometry const& geom)
{
    const auto dxinv = geom.InvCellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(F); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& Fa = F.array(mfi);
        AMREX_D_TERM(auto const& fx = liquid[0].const_array(mfi);,
                     auto const& fy = liquid[1].const_array(mfi);,
                     auto const& fz = liquid[2].const_array(mfi););
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Fa(i, j, k) += AMREX_D_TERM((fx(i, j, k) - fx(i + 1, j, k)) * dxinv[0],
                                        + (fy(i, j, k) - fy(i, j + 1, k)) * dxinv[1],
                                        + (fz(i, j, k) - fz(i, j, k + 1)) * dxinv[2]);
        });
    }
    amrex::Gpu::streamSynchronize();
}

inline void
AdvanceOwkesDesjardins (amrex::Vector<amrex::MultiFab*> const& F,
                        amrex::Vector<PLIC::FaceConstPtrs> const& umac,
                        amrex::Vector<amrex::Geometry> const& geom,
                        amrex::Vector<amrex::IntVect> const& ref_ratio,
                        amrex::Real dt, std::string const& reconstruction,
                        FillGhostCells const& fill_ghosts)
{
    const int nlevs = static_cast<int>(F.size());
    amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> liquid(nlevs);
    for (int lev = 0; lev < nlevs; ++lev) {
        liquid[lev] = make_faces(*F[lev], 1);
        PLIC::OwkesDesjardins::ComputeFluxes(liquid[lev], *F[lev], umac[lev], geom[lev], dt, reconstruction);
    }
    // Coarse faces under fine faces take the fine fluxes, so that the composite update is conservative
    for (int lev = nlevs - 1; lev > 0; --lev) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::average_down_faces(liquid[lev][dir], liquid[lev - 1][dir], ref_ratio[lev - 1], geom[lev - 1]);
        }
    }
    for (int lev = 0; lev < nlevs; ++lev) {
        UpdateFromFluxes(*F[lev], liquid[lev], geom[lev]);
    }
    for (int lev = nlevs - 1; lev > 0; --lev) {
        amrex::average_down(*F[lev], *F[lev - 1], 0, 1, ref_ratio[lev - 1]);
    }
    for (int lev = 0; lev < nlevs; ++lev) {
        fill_ghosts(lev, *F[lev]);
    }
}

//! The velocity advection term (u^{**} - u^n) / dt consistent with AdvanceOwkesDesjardins from F_old
inline void
MomentumOwkesDesjardins (amrex::Vector<amrex::MultiFab*> const& aofs,
                         amrex::Vector<amrex::MultiFab const*> const& F_old,
                         amrex::Vector<amrex::MultiFab const*> const& vel_old,
                         amrex::Vector<PLIC::FaceConstPtrs> const& umac,
                         amrex::Vector<PLIC::FaceConstPtrs> const& face_state,
                         amrex::Vector<amrex::Geometry> const& geom,
                         amrex::Vector<amrex::IntVect> const& ref_ratio,
                         amrex::Real rho_l, amrex::Real rho_g, amrex::Real dt,
                         std::string const& reconstruction)
{
    const int nlevs = static_cast<int>(aofs.size());
    amrex::Vector<amrex::MultiFab> rho(nlevs);
    amrex::Vector<amrex::MultiFab> mom(nlevs);
    amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> mflux(nlevs);
    amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> pflux(nlevs);
    for (int lev = 0; lev < nlevs; ++lev) {
        rho[lev].define(F_old[lev]->boxArray(), F_old[lev]->DistributionMap(), 1, 0);
        mom[lev].define(F_old[lev]->boxArray(), F_old[lev]->DistributionMap(), AMREX_SPACEDIM, 0);
        auto const& F_ma = F_old[lev]->const_arrays();
        auto const& rho_ma = rho[lev].arrays();
        auto const& mom_ma = mom[lev].arrays();
        auto const& vel_ma = vel_old[lev]->const_arrays();
        amrex::ParallelFor(rho[lev], [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            const amrex::Real r = PLIC::mixture(F_ma[box_no](i, j, k), rho_l, rho_g);
            rho_ma[box_no](i, j, k) = r;
            for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                mom_ma[box_no](i, j, k, n) = r * vel_ma[box_no](i, j, k, n);
            }
        });
        amrex::Gpu::streamSynchronize();

        mflux[lev] = make_faces(*F_old[lev], 1);
        pflux[lev] = make_faces(*F_old[lev], AMREX_SPACEDIM);
        PLIC::OwkesDesjardins::ComputeMomentumFluxes(mflux[lev], pflux[lev], *F_old[lev], *vel_old[lev], 0,
                                                     umac[lev], face_state[lev], geom[lev], dt, rho_l, rho_g,
                                                     reconstruction);
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
        for (int lev = nlevs - 1; lev > 0; --lev) {
            amrex::average_down_faces(mflux[lev][dir], mflux[lev - 1][dir], ref_ratio[lev - 1], geom[lev - 1]);
            amrex::average_down_faces(pflux[lev][dir], pflux[lev - 1][dir], ref_ratio[lev - 1], geom[lev - 1]);
        }
        for (int lev = 0; lev < nlevs; ++lev) {
            const amrex::Real dxinv = geom[lev].InvCellSize(dir);
            auto const& rho_ma = rho[lev].arrays();
            auto const& mom_ma = mom[lev].arrays();
            auto const& cm_ma = mflux[lev][dir].const_arrays();
            auto const& cp_ma = pflux[lev][dir].const_arrays();
            amrex::ParallelFor(rho[lev], [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
            {
                amrex::ignore_unused(k);
                const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
                const amrex::IntVect ivp = iv + e;
                rho_ma[box_no](iv) -= (cm_ma[box_no](ivp) - cm_ma[box_no](iv)) * dxinv;
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    mom_ma[box_no](iv, n) -= (cp_ma[box_no](ivp, n) - cp_ma[box_no](iv, n)) * dxinv;
                }
            });
        }
        amrex::Gpu::streamSynchronize();
    }

    for (int lev = 0; lev < nlevs; ++lev) {
        auto const& aofs_ma = aofs[lev]->arrays();
        auto const& rho_ma = rho[lev].const_arrays();
        auto const& mom_ma = mom[lev].const_arrays();
        auto const& vel_ma = vel_old[lev]->const_arrays();
        amrex::ParallelFor(rho[lev], amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
        {
            aofs_ma[box_no](i, j, k, n) = (mom_ma[box_no](i, j, k, n) / rho_ma[box_no](i, j, k) - vel_ma[box_no](i, j, k, n)) / dt;
        });
    }
    amrex::Gpu::streamSynchronize();
}

// -------------------------------------------------------------------------------------- selection

//! Advance F on all levels by one step with the scheme named by plic.advection
inline void
Advance (std::string const& advection_type,
         amrex::Vector<amrex::MultiFab*> const& F,
         amrex::Vector<PLIC::FaceConstPtrs> const& umac,
         amrex::Vector<amrex::Geometry> const& geom,
         amrex::Vector<amrex::IntVect> const& ref_ratio,
         amrex::Real dt, int step, std::string const& reconstruction,
         FillGhostCells const& fill_ghosts)
{
    amrex::ignore_unused(step);
    if (advection_type == "owkes_desjardins") {
        AdvanceOwkesDesjardins(F, umac, geom, ref_ratio, dt, reconstruction, fill_ghosts);
    } else {
        amrex::Abort("Unknown plic.advection: " + advection_type + " (expected owkes_desjardins)");
    }
}

} // namespace drivers

#endif
