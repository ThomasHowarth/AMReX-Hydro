/**
 * \file hydro_plic_weymouth_yue.cpp
 * \addtogroup PLIC
 * @{
 */

#include <hydro_plic_weymouth_yue.H>
#include <hydro_plic_weymouth_yue_K.H>

using namespace amrex::literals;

namespace PLIC::WeymouthYue {

void
ComputeSweepFluxes (amrex::MultiFab& flux, amrex::MultiFab const& F, amrex::MultiFab const& umac,
                    amrex::Geometry const& geom, int dir, amrex::Real dt,
                    std::string const& reconstruction)
{
    BL_PROFILE("PLIC::WeymouthYue::ComputeSweepFluxes()");
    CheckMesh(F, geom);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(F.nGrow() >= nGrowF(reconstruction), "PLIC: F needs PLIC::nGrowF() ghost cells");
    AMREX_ALWAYS_ASSERT(flux.boxArray() == amrex::convert(F.boxArray(), amrex::IntVect::TheDimensionVector(dir)));

    const amrex::Real dtdx = dt * geom.InvCellSize(dir);
    amrex::MultiFab plane(F.boxArray(), F.DistributionMap(), 4, 1);
    FitPlanes(plane, F, 1, reconstruction);
    auto const& flux_ma = flux.arrays();
    auto const& umac_ma = umac.const_arrays();
    auto const& F_ma = F.const_arrays();
    auto const& pl_ma = plane.const_arrays();
    amrex::ParallelFor(flux, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        const amrex::Real u = umac_ma[box_no](i, j, k);
        flux_ma[box_no](i, j, k) = u * dt * swept_fraction(i, j, k, dir, u * dtdx, F_ma[box_no], pl_ma[box_no]);
    });
    amrex::Gpu::streamSynchronize();
}

void
SweepUpdate (amrex::MultiFab& F, amrex::MultiFab const& flux, amrex::MultiFab const& umac,
             amrex::MultiFab const& cmask, amrex::Geometry const& geom, int dir, amrex::Real dt,
             amrex::MultiFab* clipped)
{
    BL_PROFILE("PLIC::WeymouthYue::SweepUpdate()");

    amrex::MultiFab local;
    if (clipped == nullptr) {
        local.define(F.boxArray(), F.DistributionMap(), 1, 0);
        clipped = &local;
    }
    const amrex::Real dxinv = geom.InvCellSize(dir);
    const amrex::Real dtdx = dt * dxinv;
    const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
    auto const& flux_ma = flux.const_arrays();
    auto const& umac_ma = umac.const_arrays();
    auto const& c_ma = cmask.const_arrays();
    auto const& F_ma = F.arrays();
    auto const& clip_ma = clipped->arrays();
    amrex::ParallelFor(*clipped, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        amrex::ignore_unused(k);
        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
        const amrex::IntVect ivp = iv + e;
        const amrex::Real a_lo = umac_ma[box_no](iv) * dtdx;
        const amrex::Real a_hi = umac_ma[box_no](ivp) * dtdx;
        const amrex::Real f = F_ma[box_no](iv) + ((flux_ma[box_no](iv) - flux_ma[box_no](ivp)) * dxinv +
                                                  c_ma[box_no](iv) * (a_hi - a_lo));
        const amrex::Real fc = amrex::Clamp(f, 0.0_rt, 1.0_rt);
        clip_ma[box_no](iv) = f - fc;
        F_ma[box_no](iv) = fc;
    });
    amrex::Gpu::streamSynchronize();
}

void
ComputeSweepMomentumFluxes (amrex::MultiFab& mass, amrex::MultiFab& momentum,
                            amrex::MultiFab const& liquid, amrex::MultiFab const& F_old,
                            amrex::MultiFab const& velocity, amrex::MultiFab const& umac,
                            amrex::MultiFab const& face_state, int dir, amrex::Real dt,
                            amrex::Real rho_l, amrex::Real rho_g)
{
    BL_PROFILE("PLIC::WeymouthYue::ComputeSweepMomentumFluxes()");
    AMREX_ALWAYS_ASSERT(F_old.nGrow() >= 2 && velocity.nGrow() >= 1);

    const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
    auto const& m_ma = mass.arrays();
    auto const& p_ma = momentum.arrays();
    auto const& l_ma = liquid.const_arrays();
    auto const& face_ma = face_state.const_arrays();
    auto const& umac_ma = umac.const_arrays();
    auto const& F_ma = F_old.const_arrays();
    auto const& vel_ma = velocity.const_arrays();
    amrex::ParallelFor(mass, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
        const amrex::IntVect ivm = iv - e;
        const amrex::Real m = rho_g * umac_ma[box_no](iv) * dt + (rho_l - rho_g) * l_ma[box_no](iv);
        m_ma[box_no](iv) = m;
        const bool near = interface_band(i, j, k, F_ma[box_no]) ||
                          interface_band(i - e[0], j - e[1], k - AMREX_D_PICK(0, 0, e[2]), F_ma[box_no]);
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            amrex::Real uf = face_ma[box_no](iv, n);
            if (near) {
                uf = (umac_ma[box_no](iv) >= 0.0_rt) ? vel_ma[box_no](ivm, n) : vel_ma[box_no](iv, n);
            }
            p_ma[box_no](iv, n) = m * uf;
        }
    });
    amrex::Gpu::streamSynchronize();
}

void
SweepMomentumUpdate (amrex::MultiFab& rho, amrex::MultiFab& momentum, amrex::MultiFab& velocity,
                     amrex::MultiFab const& mass_flux, amrex::MultiFab const& momentum_flux,
                     amrex::MultiFab const& umac, amrex::MultiFab const& cmask,
                     amrex::MultiFab const& vel_c, amrex::Geometry const& geom, int dir,
                     amrex::Real dt, amrex::Real rho_l, amrex::Real rho_g)
{
    BL_PROFILE("PLIC::WeymouthYue::SweepMomentumUpdate()");

    const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
    const amrex::Real dxinv = geom.InvCellSize(dir);
    const amrex::Real dtdx = dt * dxinv;
    auto const& m_ma = mass_flux.const_arrays();
    auto const& p_ma = momentum_flux.const_arrays();
    auto const& umac_ma = umac.const_arrays();
    auto const& c_ma = cmask.const_arrays();
    auto const& rho_ma = rho.arrays();
    auto const& mom_ma = momentum.arrays();
    auto const& vel_ma = velocity.arrays();
    auto const& velc_ma = vel_c.const_arrays();
    amrex::ParallelFor(rho, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        amrex::ignore_unused(k);
        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
        const amrex::IntVect ivp = iv + e;
        const amrex::Real dil = (umac_ma[box_no](ivp) - umac_ma[box_no](iv)) * dtdx;
        const amrex::Real rho_c = (c_ma[box_no](iv) > 0.5_rt) ? rho_l : rho_g;
        const amrex::Real r = rho_ma[box_no](iv) - (m_ma[box_no](ivp) - m_ma[box_no](iv)) * dxinv + rho_c * dil;
        rho_ma[box_no](iv) = r;
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            const amrex::Real mn = mom_ma[box_no](iv, n) - (p_ma[box_no](ivp, n) - p_ma[box_no](iv, n)) * dxinv +
                                   rho_c * velc_ma[box_no](iv, n) * dil;
            mom_ma[box_no](iv, n) = mn;
            vel_ma[box_no](iv, n) = mn / r;
        }
    });
    amrex::Gpu::streamSynchronize();
}

} // namespace PLIC::WeymouthYue
/** @}*/
