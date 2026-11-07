/**
 * \file hydro_plic_owkes_desjardins.H
 *
 * The unsplit geometric fluxes of Owkes & Desjardins (JCP 270, 2014): each face carries the liquid
 * inside its flux volume, built from the mesh nodes traced back along the flow over the step. With
 * one update per step,
 *     F <- F + sum_dir (liquid_lo - liquid_hi) / dx
 * the liquid volume is conserved to round-off without clipping. The scheme requires a face Courant
 * number below 1, and the interface away from coarse/fine boundaries, since the flux volumes do not
 * tile across them (see PLIC::TagInterface). On several levels, the coarse faces under fine faces take
 * the fine fluxes: averaged down with one time step, or refluxed with subcycling.
 *
 * The consistent momentum advection (Rudman, 1998) carries with each face the mass m_f that its flux
 * volume carries, so that
 *     rho' = rho - div(m_f),  (rho u)' = (rho u) - div(m_f u_f)
 * gives the mixture density of the new F, and a uniform velocity stays uniform. Near the interface the
 * momentum flux comes from the cuts of the flux volumes, each piece carrying the velocity of the cell
 * it lies in, and away from it u_f is the given face state.
 *
 * \addtogroup PLIC
 * @{
 */

#ifndef HYDRO_PLIC_OWKES_DESJARDINS_H
#define HYDRO_PLIC_OWKES_DESJARDINS_H

#include <hydro_plic.H>

namespace PLIC::OwkesDesjardins {

/**
 * \brief The liquid crossing each face per unit face area over the step, on one level. Faces shared by
 *        boxes have one value.
 *
 * \param liquid          output, on the faces of the grids of F, without ghost faces
 * \param F               volume fraction at t^n, with nGrowF(reconstruction) filled ghost cells
 * \param umac            face velocities at t^{n+1/2}, with one filled ghost layer
 * \param geom            geometry of this level
 * \param dt              the time step
 * \param reconstruction  "youngs", "myc" or "myc_height"
 */
void ComputeFluxes (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& liquid,
                    amrex::MultiFab const& F, FaceConstPtrs const& umac,
                    amrex::Geometry const& geom, amrex::Real dt,
                    std::string const& reconstruction);

/**
 * \brief The mass and momentum crossing each face per unit face area over the step, on one level,
 *        consistent with the liquid fluxes of ComputeFluxes for the same F, umac and dt. Faces shared
 *        by boxes have one value.
 *
 * \param mass            output, 1 component, on the faces of the grids of F
 * \param momentum        output, AMREX_SPACEDIM components, on the faces of the grids of F
 * \param F               volume fraction at t^n, as given to ComputeFluxes
 * \param velocity        cell-centered velocity at t^n, with one filled ghost layer
 * \param vel_comp        component of its first velocity component
 * \param umac            the face velocities given to ComputeFluxes
 * \param face_state      velocity at the faces (AMREX_SPACEDIM components) to use away from the interface
 * \param geom            geometry of this level
 * \param dt              the time step
 * \param rho_l           liquid density
 * \param rho_g           gas density
 * \param reconstruction  the reconstruction given to ComputeFluxes
 */
void ComputeMomentumFluxes (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& mass,
                            amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& momentum,
                            amrex::MultiFab const& F, amrex::MultiFab const& velocity, int vel_comp,
                            FaceConstPtrs const& umac, FaceConstPtrs const& face_state,
                            amrex::Geometry const& geom, amrex::Real dt,
                            amrex::Real rho_l, amrex::Real rho_g,
                            std::string const& reconstruction);

// Internal. In the header because functions with device lambdas need external linkage.

/** \brief The fluxes of one level, near the interface from the cuts of the flux volumes. */
void LevelFluxes (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& liquid,
                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>* mass_momentum,
                  amrex::iMultiFab const& band,
                  amrex::MultiFab const& F, amrex::MultiFab const& plane,
                  FaceConstPtrs const& umac, amrex::Geometry const& geom, amrex::Real dt,
                  amrex::MultiFab const* velocity, int vel_comp,
                  amrex::Real rho_l, amrex::Real rho_g);

/**
 * \brief Mark the cells within one cell of the interface.
 *
 * \param band  output, 1 near the interface, on the valid cells and ngrow ghost layers
 * \param F     volume fraction with ngrow + 1 filled ghost cells
 */
void InterfaceBand (amrex::iMultiFab& band, amrex::MultiFab const& F, int ngrow);

} // namespace PLIC::OwkesDesjardins

#endif
/** @}*/
