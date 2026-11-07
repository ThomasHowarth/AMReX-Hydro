#ifndef PLIC_DRIVERS_H_
#define PLIC_DRIVERS_H_

// How a solver advances the volume fraction with the operators of the PLIC library: here all levels
// with the same time step. A subcycling solver advances each level on its own instead, and refluxes
// with the same fluxes.

#include <hydro_plic.H>

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
    amrex::ignore_unused(F, umac, geom, ref_ratio, dt, step, reconstruction, fill_ghosts);
    amrex::Abort("Unknown plic.advection: " + advection_type);
}

} // namespace drivers

#endif
