/**
 * \file hydro_plic_weymouth_yue_K.H
 *
 * Kernels of the directionally split advection of Weymouth & Yue (JCP 229, 2010).
 *
 * \addtogroup PLIC
 *  @{
 */

#ifndef HYDRO_PLIC_WEYMOUTH_YUE_K_H
#define HYDRO_PLIC_WEYMOUTH_YUE_K_H

#include <hydro_plic_K.H>

namespace PLIC {

/**
 * \brief Liquid fraction of the volume swept through the low face of cell (i,j,k) in direction dir
 *        during one split sweep, given the face Courant number a = u_face * dt / dx. The swept
 *        region lies in the upwind cell.
 *
 * \param plane the PLIC plane of each mixed cell (m_x, m_y, m_z, alpha)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
swept_fraction(const int i,
               const int j,
               const int k,
               const int dir,
               const amrex::Real a,
               amrex::Array4<amrex::Real const> const& F,
               amrex::Array4<amrex::Real const> const& plane) {
    amrex::ignore_unused(k);
    if (a == 0.0_rt) {
        return 0.0_rt;
    }
    const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
    const amrex::IntVect up = (a > 0.0_rt) ? iv - amrex::IntVect::TheDimensionVector(dir) : iv;
    const amrex::Real Fu = F(up);
    if (Fu <= pure_tol) {
        return 0.0_rt;
    }
    if (Fu >= 1.0_rt - pure_tol) {
        return 1.0_rt;
    }

    const amrex::Real mx = plane(up, 0);
    const amrex::Real my = plane(up, 1);
    const amrex::Real mz = plane(up, 2);
    const amrex::Real alpha = plane(up, 3);

    // Positive velocity sweeps the high end [1-a, 1] of the upwind cell, negative velocity the low
    // end [0, -a]
    const amrex::Real r0 = (a > 0.0_rt) ? 1.0_rt - a : 0.0_rt;
    const amrex::Real dr0 = std::abs(a);
    if (dir == 0) {
        return cut_volume(mx, my, mz, alpha, r0, dr0);
    }
    if (dir == 1) {
        return cut_volume(my, mz, mx, alpha, r0, dr0);
    }
    return cut_volume(mz, mx, my, alpha, r0, dr0);
}

} // namespace PLIC

#endif
/** @}*/
