/**
 * \file hydro_plic.cpp
 * \addtogroup PLIC
 * @{
 */

#include <hydro_plic.H>

#include <AMReX_MultiFabUtil.H>

#include <limits>

using namespace amrex::literals;

namespace PLIC {

// Neither the plane geometry nor the flux volumes account for cut cells or for R-Z volumes
void CheckMesh (amrex::MultiFab const& F, amrex::Geometry const& geom)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom.IsCartesian(), "PLIC requires Cartesian coordinates");
#ifdef AMREX_USE_EB
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!F.hasEBFabFactory(), "PLIC does not support embedded boundaries");
#else
    amrex::ignore_unused(F);
#endif
}

void
SharpFaceDensity (FacePtrs const& rho_face,
                  amrex::Vector<amrex::MultiFab const*> const& F,
                  amrex::Geometry const& geom,
                  amrex::Real rho_l, amrex::Real rho_g,
                  std::string const& reconstruction)
{
    BL_PROFILE("PLIC::SharpFaceDensity()");
    AMREX_ALWAYS_ASSERT(!F.empty());

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        rho_face[dir]->setVal(0.0_rt);
    }

    // Segment liquid fraction, averaged over the given F
    const amrex::Real weight = 1.0_rt / static_cast<amrex::Real>(F.size());
    amrex::MultiFab plane(F[0]->boxArray(), F[0]->DistributionMap(), 4, 1);
    for (auto const* Fi : F) {
        CheckMesh(*Fi, geom);
        // The planes of the cells on either side of each face
        FitPlanes(plane, *Fi, 1, reconstruction);
        auto const& F_ma = Fi->const_arrays();
        auto const& pl_ma = plane.const_arrays();
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            auto const& fc_ma = rho_face[dir]->arrays();
            amrex::ParallelFor(*rho_face[dir], [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
            {
                fc_ma[box_no](i, j, k) += weight * face_segment_fraction(i, j, k, dir, F_ma[box_no], pl_ma[box_no]);
            });
        }
        amrex::Gpu::streamSynchronize();
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto const& fc_ma = rho_face[dir]->arrays();
        amrex::ParallelFor(*rho_face[dir], [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            fc_ma[box_no](i, j, k) = rho_g + (rho_l - rho_g) * fc_ma[box_no](i, j, k);
        });
        amrex::Gpu::streamSynchronize();
        // A face must have one coefficient, or the projection operator is neither conservative nor
        // symmetric. Keep one copy against round-off.
        rho_face[dir]->OverrideSync(geom.periodicity());
    }
}

void
TagInterface (amrex::TagBoxArray& tags, amrex::MultiFab const& F, int nband)
{
    AMREX_ALWAYS_ASSERT(F.nGrow() >= nband);
    auto const& tag_ma = tags.arrays();
    auto const& F_ma = F.const_arrays();
    amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        if (interface_band(i, j, k, F_ma[box_no], nband)) {
            tag_ma[box_no](i, j, k) = amrex::TagBox::SET;
        }
    });
    amrex::Gpu::streamSynchronize();
}

void
FillMixture (amrex::MultiFab& dst, int dcomp, amrex::MultiFab const& F, int ngrow,
             amrex::Real liq, amrex::Real gas, bool harmonic)
{
    AMREX_ALWAYS_ASSERT(dst.nGrow() >= ngrow && F.nGrow() >= ngrow);
    auto const& dst_ma = dst.arrays();
    auto const& F_ma = F.const_arrays();
    const int iharm = harmonic ? 1 : 0;
    amrex::ParallelFor(dst, amrex::IntVect(ngrow),
    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        dst_ma[box_no](i, j, k, dcomp) = mixture(F_ma[box_no](i, j, k), liq, gas, iharm);
    });
    amrex::Gpu::streamSynchronize();
}

amrex::Real
LiquidVolume (amrex::Vector<amrex::MultiFab const*> const& F,
              amrex::Vector<amrex::Geometry> const& geom,
              amrex::Vector<amrex::IntVect> const& ref_ratio)
{
    amrex::Real volume = 0.0_rt;
    const int nlevs = static_cast<int>(F.size());
    for (int lev = 0; lev < nlevs; ++lev) {
        amrex::iMultiFab mask;
        if (lev < nlevs - 1) {
            mask = amrex::makeFineMask(*F[lev], F[lev + 1]->boxArray(), ref_ratio[lev], 1, 0);
        } else {
            mask.define(F[lev]->boxArray(), F[lev]->DistributionMap(), 1, 0);
            mask.setVal(1);
        }
        auto const& F_ma = F[lev]->const_arrays();
        auto const& mask_ma = mask.const_arrays();
        amrex::Real sum = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum>{}, amrex::TypeList<amrex::Real>{}, *F[lev],
            amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> amrex::GpuTuple<amrex::Real>
            {
                return {(mask_ma[box_no](i, j, k) != 0) ? F_ma[box_no](i, j, k) : 0.0_rt};
            });
        amrex::ParallelDescriptor::ReduceRealSum(sum);
        volume += sum * geom[lev].CellSize(0) * AMREX_D_TERM(1.0_rt, *geom[lev].CellSize(1), *geom[lev].CellSize(2));
    }
    return volume;
}

amrex::Real
EstimateDt (amrex::Vector<amrex::MultiFab const*> const& vel, int vel_comp,
            amrex::Vector<amrex::Geometry> const& geom, amrex::Real cfl)
{
    amrex::Real estdt = std::numeric_limits<amrex::Real>::max();
    for (int lev = 0; lev < static_cast<int>(vel.size()); ++lev) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const amrex::Real umax = vel[lev]->norm0(vel_comp + dir, 0, true);
            if (umax > 0.0_rt) {
                estdt = amrex::min(estdt, cfl * geom[lev].CellSize(dir) / umax);
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealMin(estdt);
    return estdt;
}

} // namespace PLIC
/** @}*/
