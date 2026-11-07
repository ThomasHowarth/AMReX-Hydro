/**
 * \file hydro_plic.H
 *
 * Geometric volume-of-fluid transport with a piecewise-linear interface calculation (PLIC), for any
 * AMReX-based solver. It works on the solver's MultiFabs and holds no solver state.
 *
 * Like the rest of AMReX-Hydro it provides spatial operators on one level, and leaves the time
 * integration to the solver: the update, the ghost cells, the synchronization of the levels (average
 * down or reflux) and any subcycling. The reconstruction, how the plane of each mixed cell is fitted,
 * is selected by name: "youngs", "myc" or "myc_height". Each advection scheme has its own namespace
 * and header, with the fluxes and the operators of its time integration.
 *
 * Volume fractions F are the liquid fraction of each cell. Requires Cartesian coordinates, and has no
 * embedded boundary support.
 *
 * \addtogroup PLIC
 *  @{
 */

#ifndef HYDRO_PLIC_H
#define HYDRO_PLIC_H

#include <hydro_plic_K.H>

#include <AMReX_Array.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_TagBox.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include <string>

namespace PLIC {

/** \brief Pointers to a face-centered quantity, one MultiFab per direction. */
using FacePtrs = amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>;
/** \brief Const pointers to a face-centered quantity, one MultiFab per direction. */
using FaceConstPtrs = amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM>;

/**
 * \brief The ghost cells of F that the operators of this library read, with the given
 *        reconstruction. Allocate and fill at least this many.
 */
[[nodiscard]] int nGrowF (std::string const& reconstruction);

/**
 * \brief Abort unless the mesh is supported: Cartesian coordinates and no embedded boundaries.
 */
void CheckMesh (amrex::MultiFab const& F, amrex::Geometry const& geom);

/**
 * \brief Fit the plane of each mixed cell, and zero the others.
 *
 * \param plane           4 components (m_x, m_y, m_z, alpha), in unit-cube cell coordinates, with the
 *                        liquid on the side m.x <= alpha and sum|m| = 1
 * \param F               volume fraction, with its ghost cells filled
 * \param ngrow           ghost layers of plane to fit
 * \param reconstruction  "youngs", "myc" or "myc_height"
 */
void FitPlanes (amrex::MultiFab& plane, amrex::MultiFab const& F, int ngrow,
                std::string const& reconstruction);

/**
 * \brief Overwrite the ghost cells of F beyond the domain faces where F is extrapolated, by extending
 *        the planes of the interior interface. At a wall the interface then meets the wall at whatever
 *        angle it has: a free contact line. Extrapolated values impose 90 degrees instead, and the
 *        inconsistent normals near the wall unbalance gravity at a tilted interface. At an outflow the
 *        interface continues smoothly. The result does not depend on the box layout.
 *
 * The boundary condition of F on each domain face selects what happens there:
 *  - foextrap, hoextrap: the ghost cells are overwritten (a wall or an outflow);
 *  - ext_dir: the ghost cells are kept, as the volume fraction that flows in;
 *  - reflect_even: the ghost cells are kept, as the mirror image of a symmetry plane (90 degrees);
 *  - int_dir: periodic, nothing to do;
 *  - reflect_odd is not a valid condition for a volume fraction, and aborts.
 *
 * \param F               volume fraction with all its ghost cells filled, including those beyond the
 *                        domain (first-order extrapolation is enough at foextrap and hoextrap faces)
 *                        and at coarse/fine boundaries
 * \param geom            geometry of this level
 * \param bcrec           boundary condition of F, as for its ghost cells
 * \param reconstruction  "youngs", "myc" or "myc_height"
 */
void FillWallGhosts (amrex::MultiFab& F, amrex::Geometry const& geom, amrex::BCRec const& bcrec,
                     std::string const& reconstruction);

/**
 * \brief Face density rho_g + (rho_l - rho_g) * <face_segment_fraction>, with the average over the
 *        given volume fractions. For a planar interface normal to gravity it makes rho_face * g a
 *        discrete gradient, which is the hydrostatic balance of a balanced-force projection (Francois
 *        et al., JCP 213, 2006). Each face gets one value.
 *
 * \param rho_face        output, face-centered with any ghost cells, which are not set
 * \param F               the volume fractions to average, e.g. at t^n and t^{n+1} for t^{n+1/2}, with
 *                        nGrowF() filled ghost cells
 * \param geom            geometry of this level
 * \param rho_l           liquid density
 * \param rho_g           gas density
 * \param reconstruction  "youngs", "myc" or "myc_height"
 */
void SharpFaceDensity (FacePtrs const& rho_face,
                       amrex::Vector<amrex::MultiFab const*> const& F,
                       amrex::Geometry const& geom,
                       amrex::Real rho_l, amrex::Real rho_g,
                       std::string const& reconstruction);

/**
 * \brief Tag the cells within nband cells of the interface.
 *
 * Keep the interface on the finest level: the flux volumes of a geometric unsplit scheme do not tile
 * across coarse/fine boundaries, which must stay in pure cells.
 *
 * \param F  volume fraction with nband filled ghost cells
 */
void TagInterface (amrex::TagBoxArray& tags, amrex::MultiFab const& F, int nband = 2);

/**
 * \brief Mixture property (density, viscosity) from the volume fraction: an arithmetic or harmonic
 *        volume average of the phase values.
 *
 * \param dst       output
 * \param dcomp     component of dst
 * \param F         volume fraction, with ngrow filled ghost cells
 * \param ngrow     ghost layers to fill
 * \param liq       value in the liquid
 * \param gas       value in the gas
 * \param harmonic  harmonic rather than arithmetic average
 */
void FillMixture (amrex::MultiFab& dst, int dcomp, amrex::MultiFab const& F, int ngrow,
                  amrex::Real liq, amrex::Real gas, bool harmonic = false);

/**
 * \brief The liquid volume on the composite grid, not counting covered cells.
 */
[[nodiscard]] amrex::Real LiquidVolume (amrex::Vector<amrex::MultiFab const*> const& F,
                                        amrex::Vector<amrex::Geometry> const& geom,
                                        amrex::Vector<amrex::IntVect> const& ref_ratio);

/**
 * \brief The largest time step for a Courant number cfl, from the cell-centered velocity.
 *
 * \note The advection uses the face velocity at t^{n+1/2}, which can be larger: keep cfl well below 1.
 *
 * \param vel       cell-centered velocity on each level
 * \param vel_comp  component of its first velocity component
 */
[[nodiscard]] amrex::Real EstimateDt (amrex::Vector<amrex::MultiFab const*> const& vel, int vel_comp,
                                      amrex::Vector<amrex::Geometry> const& geom, amrex::Real cfl);

/**
 * \brief Prescribed face velocities u = curl(A), on all the faces of umac including ghost faces.
 *        They are discretely divergence-free to round-off and consistent across coarse/fine faces,
 *        which makes them suited to verifying a scheme. See face_velocity_from_potential.
 *
 * \param potential potential(x, time, A), callable on the device
 */
template <typename Potential>
void FaceVelocityFromPotential (FacePtrs const& umac, amrex::Geometry const& geom, amrex::Real time,
                                Potential const& potential)
{
    const auto geomdata = geom.data();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto const& um = umac[dir]->arrays();
        amrex::ParallelFor(*umac[dir], umac[dir]->nGrowVect(),
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            um[box_no](i, j, k) = face_velocity_from_potential(
                dir, amrex::IntVect(AMREX_D_DECL(i, j, k)), time, geomdata, potential);
        });
    }
    amrex::Gpu::streamSynchronize();
}

} // namespace PLIC

#endif
/** @}*/
