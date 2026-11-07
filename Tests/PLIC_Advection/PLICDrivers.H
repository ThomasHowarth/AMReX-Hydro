#ifndef PLIC_DRIVERS_H_
#define PLIC_DRIVERS_H_

// How a solver advances the volume fraction with the operators of the PLIC library: here all levels
// with the same time step. A subcycling solver advances each level on its own instead, and refluxes
// with the same fluxes.

#include <hydro_plic.H>
#include <hydro_plic_owkes_desjardins.H>
#include <hydro_plic_weymouth_yue.H>

#include <AMReX_MultiFabUtil.H>

#include <cmath>
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

// ----------------------------------------------------------------------------------- weymouth_yue

//! What the momentum advection replays of a step
struct WeymouthYueRecord
{
    amrex::Real dt_sub = 0.0;
    amrex::Vector<int> sweep_dir;
    amrex::Vector<amrex::Vector<amrex::MultiFab>> sweep_flux; //!< for each sweep and level
    amrex::Vector<amrex::Vector<amrex::MultiFab>> cmask;      //!< for each sub-step and level
};

// Clipping only compensates round-off when the sweeps are bounded (per-sweep Courant number of at most
// max_courant, divergence-free umac): report anything larger than this
#ifdef AMREX_USE_FLOAT
inline constexpr amrex::Real clip_report_tol = amrex::Real(1.0e-4);
#else
inline constexpr amrex::Real clip_report_tol = amrex::Real(1.0e-12);
#endif

//! One sweep in direction dir on all levels
inline void
SweepWeymouthYue (amrex::Vector<amrex::MultiFab*> const& F,
                  amrex::Vector<PLIC::FaceConstPtrs> const& umac,
                  amrex::Vector<amrex::Geometry> const& geom,
                  amrex::Vector<amrex::IntVect> const& ref_ratio,
                  std::string const& reconstruction, FillGhostCells const& fill_ghosts,
                  int dir, amrex::Real dt,
                  amrex::Vector<amrex::MultiFab> const& cmask,
                  amrex::Vector<amrex::MultiFab>& flux)
{
    const int nlevs = static_cast<int>(F.size());
    for (int lev = 0; lev < nlevs; ++lev) {
        flux[lev].define(amrex::convert(F[lev]->boxArray(), amrex::IntVect::TheDimensionVector(dir)),
                         F[lev]->DistributionMap(), 1, 0);
        PLIC::WeymouthYue::ComputeSweepFluxes(flux[lev], *F[lev], *umac[lev][dir], geom[lev], dir, dt, reconstruction);
    }
    // Coarse faces under fine faces take the fine-face average
    for (int lev = nlevs - 1; lev > 0; --lev) {
        amrex::average_down_faces(flux[lev], flux[lev - 1], ref_ratio[lev - 1], geom[lev - 1]);
    }
    for (int lev = 0; lev < nlevs; ++lev) {
        amrex::MultiFab clipped(F[lev]->boxArray(), F[lev]->DistributionMap(), 1, 0);
        PLIC::WeymouthYue::SweepUpdate(*F[lev], flux[lev], *umac[lev][dir], cmask[lev], geom[lev], dir, dt, &clipped);
        // Covered cells are overwritten by the average down
        if (lev < nlevs - 1) {
            const amrex::iMultiFab mask = amrex::makeFineMask(*F[lev], F[lev + 1]->boxArray(), ref_ratio[lev], 1, 0);
            auto const& clip_ma = clipped.arrays();
            auto const& mask_ma = mask.const_arrays();
            amrex::ParallelFor(clipped, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
            {
                clip_ma[box_no](i, j, k) *= static_cast<amrex::Real>(mask_ma[box_no](i, j, k));
            });
            amrex::Gpu::streamSynchronize();
        }
        const amrex::Real clip_max = clipped.norm0();
        if (clip_max > clip_report_tol) {
            const amrex::Real cellvol = AMREX_D_TERM(geom[lev].CellSize(0), *geom[lev].CellSize(1), *geom[lev].CellSize(2));
            amrex::Print() << "   PLIC: WARNING sweep " << dir << " level " << lev
                           << " clipped volume " << clipped.sum() * cellvol
                           << " (max cell " << clip_max << ")\n";
        }
    }
    for (int lev = nlevs - 1; lev > 0; --lev) {
        amrex::average_down(*F[lev], *F[lev - 1], 0, 1, ref_ratio[lev - 1]);
    }
    // The next sweep fits the planes again, with these ghost cells
    for (int lev = 0; lev < nlevs; ++lev) {
        fill_ghosts(lev, *F[lev]);
    }
}

inline void
AdvanceWeymouthYue (amrex::Vector<amrex::MultiFab*> const& F,
                    amrex::Vector<PLIC::FaceConstPtrs> const& umac,
                    amrex::Vector<amrex::Geometry> const& geom,
                    amrex::Vector<amrex::IntVect> const& ref_ratio,
                    amrex::Real dt, int step, std::string const& reconstruction,
                    FillGhostCells const& fill_ghosts,
                    WeymouthYueRecord* record = nullptr)
{
    const int nlevs = static_cast<int>(F.size());

    // Largest face velocity / dx. The step is usually estimated from the cell-centered velocity at
    // t^n, while the sweeps use the face velocity at t^{n+1/2}, which can be larger
    amrex::Real umac_rate = 0.0;
    for (int lev = 0; lev < nlevs; ++lev) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            umac_rate = amrex::max(umac_rate, umac[lev][dir]->norm0(0, 0, true) * geom[lev].InvCellSize(dir));
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(umac_rate);
    const amrex::Real courant_max = umac_rate * dt;
    const int nsub = amrex::max(1, static_cast<int>(std::ceil(courant_max / PLIC::WeymouthYue::max_courant)));
    if (nsub > 1) {
        amrex::Print() << "   PLIC: face Courant number " << courant_max
                       << " exceeds " << PLIC::WeymouthYue::max_courant << ", advecting in " << nsub << " sub-steps\n";
    }
    const amrex::Real dt_sub = dt / static_cast<amrex::Real>(nsub);
    if (record != nullptr) {
        record->dt_sub = dt_sub;
    }

    for (int isub = 0; isub < nsub; ++isub) {
        // c = H(F - 1/2) frozen over the (sub-)step
        amrex::Vector<amrex::MultiFab> cmask(nlevs);
        for (int lev = 0; lev < nlevs; ++lev) {
            cmask[lev].define(F[lev]->boxArray(), F[lev]->DistributionMap(), 1, 0);
            auto const& c_ma = cmask[lev].arrays();
            auto const& F_ma = F[lev]->const_arrays();
            amrex::ParallelFor(cmask[lev], [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
            {
                c_ma[box_no](i, j, k) = (F_ma[box_no](i, j, k) > amrex::Real(0.5)) ? amrex::Real(1.0) : amrex::Real(0.0);
            });
        }
        amrex::Gpu::streamSynchronize();

        // Cycle the order of the sweeps with the step number, so that restarts reproduce it
        const int cycle = step + isub;
        for (int s = 0; s < AMREX_SPACEDIM; ++s) {
            const int dir = (cycle + s) % AMREX_SPACEDIM;
            amrex::Vector<amrex::MultiFab> flux(nlevs);
            SweepWeymouthYue(F, umac, geom, ref_ratio, reconstruction, fill_ghosts, dir, dt_sub, cmask, flux);
            if (record != nullptr) {
                record->sweep_dir.push_back(dir);
                record->sweep_flux.push_back(std::move(flux));
            }
        }
        if (record != nullptr) {
            record->cmask.push_back(std::move(cmask));
        }
    }
}

//! The velocity advection term (u^{**} - u^n) / dt consistent with AdvanceWeymouthYue from F_old
inline void
MomentumWeymouthYue (amrex::Vector<amrex::MultiFab*> const& aofs,
                     amrex::Vector<amrex::MultiFab const*> const& F_old,
                     amrex::Vector<amrex::MultiFab const*> const& vel_old,
                     amrex::Vector<PLIC::FaceConstPtrs> const& umac,
                     amrex::Vector<PLIC::FaceConstPtrs> const& face_state,
                     amrex::Vector<amrex::Geometry> const& geom,
                     amrex::Vector<amrex::IntVect> const& ref_ratio,
                     amrex::Real rho_l, amrex::Real rho_g, amrex::Real dt,
                     WeymouthYueRecord const& record)
{
    const int nlevs = static_cast<int>(aofs.size());
    const int nsweep = static_cast<int>(record.sweep_dir.size());
    AMREX_ALWAYS_ASSERT(nsweep > 0 && nsweep == static_cast<int>(record.cmask.size()) * AMREX_SPACEDIM);

    // Mixture density, momentum and the velocity with a ghost layer for the upwinding; u_c is the
    // velocity at the start of each (sub-)step
    amrex::Vector<amrex::MultiFab> rho(nlevs);
    amrex::Vector<amrex::MultiFab> mom(nlevs);
    amrex::Vector<amrex::MultiFab> vel(nlevs);
    amrex::Vector<amrex::MultiFab> velc(nlevs);
    for (int lev = 0; lev < nlevs; ++lev) {
        amrex::BoxArray const& grids = F_old[lev]->boxArray();
        amrex::DistributionMapping const& dmap = F_old[lev]->DistributionMap();
        rho[lev].define(grids, dmap, 1, 0);
        mom[lev].define(grids, dmap, AMREX_SPACEDIM, 0);
        vel[lev].define(grids, dmap, AMREX_SPACEDIM, 1);
        velc[lev].define(grids, dmap, AMREX_SPACEDIM, 0);
        amrex::MultiFab::Copy(vel[lev], *vel_old[lev], 0, 0, AMREX_SPACEDIM, 1);
        auto const& F_ma = F_old[lev]->const_arrays();
        auto const& rho_ma = rho[lev].arrays();
        auto const& mom_ma = mom[lev].arrays();
        auto const& vel_ma = vel[lev].const_arrays();
        amrex::ParallelFor(rho[lev], [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            const amrex::Real r = PLIC::mixture(F_ma[box_no](i, j, k), rho_l, rho_g);
            rho_ma[box_no](i, j, k) = r;
            for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                mom_ma[box_no](i, j, k, n) = r * vel_ma[box_no](i, j, k, n);
            }
        });
    }
    amrex::Gpu::streamSynchronize();

    for (int isweep = 0; isweep < nsweep; ++isweep) {
        const int dir = record.sweep_dir[isweep];
        const int isub = isweep / AMREX_SPACEDIM;
        if (isweep % AMREX_SPACEDIM == 0) {
            for (int lev = 0; lev < nlevs; ++lev) {
                amrex::MultiFab::Copy(velc[lev], vel[lev], 0, 0, AMREX_SPACEDIM, 0);
            }
        }
        amrex::Vector<amrex::MultiFab> mflux(nlevs);
        amrex::Vector<amrex::MultiFab> pflux(nlevs);
        for (int lev = 0; lev < nlevs; ++lev) {
            auto const& lflux = record.sweep_flux[isweep][lev];
            mflux[lev].define(lflux.boxArray(), lflux.DistributionMap(), 1, 0);
            pflux[lev].define(lflux.boxArray(), lflux.DistributionMap(), AMREX_SPACEDIM, 0);
            PLIC::WeymouthYue::ComputeSweepMomentumFluxes(mflux[lev], pflux[lev], lflux, *F_old[lev], vel[lev],
                                                          *umac[lev][dir], *face_state[lev][dir], dir,
                                                          record.dt_sub, rho_l, rho_g);
        }
        for (int lev = nlevs - 1; lev > 0; --lev) {
            amrex::average_down_faces(pflux[lev], pflux[lev - 1], ref_ratio[lev - 1], geom[lev - 1]);
        }
        for (int lev = 0; lev < nlevs; ++lev) {
            PLIC::WeymouthYue::SweepMomentumUpdate(rho[lev], mom[lev], vel[lev], mflux[lev], pflux[lev],
                                                   *umac[lev][dir], record.cmask[isub][lev], velc[lev], geom[lev],
                                                   dir, record.dt_sub, rho_l, rho_g);
        }
        // The intermediate velocities of neighbouring boxes; coarse/fine and physical ghost cells keep
        // t^n (the interface band is refined away from coarse/fine boundaries, and the wall mass fluxes
        // vanish)
        for (int lev = 0; lev < nlevs; ++lev) {
            vel[lev].FillBoundary(0, AMREX_SPACEDIM, amrex::IntVect(1), geom[lev].periodicity());
        }
    }

    for (int lev = 0; lev < nlevs; ++lev) {
        auto const& aofs_ma = aofs[lev]->arrays();
        auto const& vel_ma = vel[lev].const_arrays();
        auto const& velo_ma = vel_old[lev]->const_arrays();
        amrex::ParallelFor(vel[lev], amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
        {
            aofs_ma[box_no](i, j, k, n) = (vel_ma[box_no](i, j, k, n) - velo_ma[box_no](i, j, k, n)) / dt;
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
    if (advection_type == "owkes_desjardins") {
        AdvanceOwkesDesjardins(F, umac, geom, ref_ratio, dt, reconstruction, fill_ghosts);
    } else if (advection_type == "weymouth_yue") {
        AdvanceWeymouthYue(F, umac, geom, ref_ratio, dt, step, reconstruction, fill_ghosts);
    } else {
        amrex::Abort("Unknown plic.advection: " + advection_type + " (expected owkes_desjardins or weymouth_yue)");
    }
}

} // namespace drivers

#endif
