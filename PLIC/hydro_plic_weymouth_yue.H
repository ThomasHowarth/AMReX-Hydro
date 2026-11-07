/**
 * \file hydro_plic_weymouth_yue.H
 *
 * The directionally split Eulerian-implicit operators of Weymouth & Yue (JCP 229, 2010). A step is
 * a sequence of sweeps, one per direction, each of which fits the planes, computes the fluxes and
 * updates
 *     F <- F + (flux_lo - flux_hi) / dx + c * (a_hi - a_lo)
 * where flux = u dt * (liquid fraction of the swept volume) is the liquid crossing the face per unit
 * face area, a = u dt / dx is the face Courant number and c = H(F - 1/2) is taken at the start of the
 * step and frozen over its sweeps. The dilatation terms c * d(a)/dx then sum to c * div(u) dt = 0 over
 * the sweeps, so the step is conservative for a discretely divergence-free umac.
 *
 * The solver owns the sequence: the order of the sweeps, which should cycle from one step to the next
 * to avoid a directional bias, the ghost cells of F and the synchronization of the levels between
 * sweeps, and sub-steps. F stays in [0,1] for a per-sweep Courant number of at most max_courant, and
 * a larger step is divided into sub-steps, each with its own c. SweepUpdate clips F to [0,1], which
 * then only removes round-off: report any more than that.
 *
 * The consistent momentum advection (Rudman, 1998, in the split form of Arrufat et al., JCP 2021)
 * replays the same sweeps on the mixture density rho and the momentum,
 *     rho'     = rho     - (m_hi - m_lo) / dx           + rho_c (a_hi - a_lo),
 *     (rho u)' = (rho u) - (m_hi u_hi - m_lo u_lo) / dx + rho_c u_c (a_hi - a_lo),
 * with the sweep fluxes of the liquid, the same c (rho_c its phase density) and u_c the velocity at the
 * start of the (sub-)step. rho' is then the mixture density of the swept F and a uniform velocity
 * stays uniform across the interface. Within one cell of the interface u_f is the upwind intermediate
 * cell velocity, which must be exchanged between boxes after each sweep: liquid that crosses a gas cell
 * within a step leaves with the velocity it came in with, while upwinding with the velocity at t^n
 * would charge it with the gas velocity, an error amplified by rho_l / rho_g. Elsewhere u_f is the
 * given face state.
 *
 * \addtogroup PLIC
 * @{
 */

#ifndef HYDRO_PLIC_WEYMOUTH_YUE_H
#define HYDRO_PLIC_WEYMOUTH_YUE_H

#include <hydro_plic.H>

namespace PLIC::WeymouthYue {

//! The largest face Courant number of a sweep for which F stays in [0,1]
constexpr amrex::Real max_courant = 0.5_rt;

/**
 * \brief The liquid crossing each face normal to dir per unit face area over a sweep, on one level.
 *
 * \param flux            output, on the faces of the grids of F normal to dir, without ghost faces
 * \param F               volume fraction before the sweep, with nGrowF(reconstruction) filled ghost cells
 * \param umac            face velocity normal to dir, with one filled ghost layer
 * \param geom            geometry of this level
 * \param dir             direction of the sweep
 * \param dt              the time step of the sweep
 * \param reconstruction  "youngs", "myc" or "myc_height"
 */
void ComputeSweepFluxes (amrex::MultiFab& flux, amrex::MultiFab const& F, amrex::MultiFab const& umac,
                         amrex::Geometry const& geom, int dir, amrex::Real dt,
                         std::string const& reconstruction);

/**
 * \brief The update of a sweep, on the valid cells of one level, clipped to [0,1].
 *
 * \param F        volume fraction
 * \param flux     the fluxes of ComputeSweepFluxes, synchronized with the other levels
 * \param umac     face velocity normal to dir
 * \param cmask    c = H(F - 1/2) at the start of the (sub-)step
 * \param geom     geometry of this level
 * \param dir      direction of the sweep
 * \param dt       the time step of the sweep
 * \param clipped  if not null, the volume fraction removed by the clipping in each cell
 */
void SweepUpdate (amrex::MultiFab& F, amrex::MultiFab const& flux, amrex::MultiFab const& umac,
                  amrex::MultiFab const& cmask, amrex::Geometry const& geom, int dir, amrex::Real dt,
                  amrex::MultiFab* clipped = nullptr);

/**
 * \brief The mass and momentum crossing each face normal to dir per unit face area in a sweep, on one
 *        level.
 *
 * \param mass        output, 1 component, on the faces of the grids normal to dir
 * \param momentum    output, AMREX_SPACEDIM components, on the same faces
 * \param liquid      the fluxes of the sweep of F, synchronized with the other levels
 * \param F_old       volume fraction at t^n, with two filled ghost cells, which sets the interface band
 * \param velocity    cell-centered velocity before this sweep, components 0 to AMREX_SPACEDIM - 1, with one
 *                    filled ghost layer
 * \param umac        face velocity normal to dir
 * \param face_state  velocity at the faces (AMREX_SPACEDIM components) to use away from the interface
 * \param dir         direction of the sweep
 * \param dt          the time step of the sweep
 * \param rho_l       liquid density
 * \param rho_g       gas density
 */
void ComputeSweepMomentumFluxes (amrex::MultiFab& mass, amrex::MultiFab& momentum,
                                 amrex::MultiFab const& liquid, amrex::MultiFab const& F_old,
                                 amrex::MultiFab const& velocity, amrex::MultiFab const& umac,
                                 amrex::MultiFab const& face_state, int dir, amrex::Real dt,
                                 amrex::Real rho_l, amrex::Real rho_g);

/**
 * \brief The update of a sweep of the momentum advection, on the valid cells of one level.
 *
 * \param rho            mixture density
 * \param momentum       momentum, AMREX_SPACEDIM components
 * \param velocity       output, momentum / rho on the valid cells
 * \param mass_flux      the mass fluxes of ComputeSweepMomentumFluxes
 * \param momentum_flux  the momentum fluxes of ComputeSweepMomentumFluxes, synchronized with the other levels
 * \param umac           face velocity normal to dir
 * \param cmask          c = H(F - 1/2) at the start of the (sub-)step
 * \param vel_c          velocity at the start of the (sub-)step
 * \param geom           geometry of this level
 * \param dir            direction of the sweep
 * \param dt             the time step of the sweep
 * \param rho_l          liquid density
 * \param rho_g          gas density
 */
void SweepMomentumUpdate (amrex::MultiFab& rho, amrex::MultiFab& momentum, amrex::MultiFab& velocity,
                          amrex::MultiFab const& mass_flux, amrex::MultiFab const& momentum_flux,
                          amrex::MultiFab const& umac, amrex::MultiFab const& cmask,
                          amrex::MultiFab const& vel_c, amrex::Geometry const& geom, int dir,
                          amrex::Real dt, amrex::Real rho_l, amrex::Real rho_g);

} // namespace PLIC::WeymouthYue

#endif
/** @}*/
