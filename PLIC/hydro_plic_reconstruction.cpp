/**
 * \file hydro_plic_reconstruction.cpp
 * \addtogroup PLIC
 * @{
 */

#include <hydro_plic.H>
#include <hydro_plic_normals_K.H>

#include <AMReX_BC_TYPES.H>
#include <AMReX_LayoutData.H>
#include <AMReX_ParReduce.H>

#include <exception>
#include <string>
#include <utility>

using namespace amrex::literals;

namespace PLIC {

namespace {

//! The domain faces whose ghost cells FillWallGhosts overwrites: extend[dir][0] low, extend[dir][1] high
using ExtendFlags = amrex::Array<amrex::Array<bool, 2>, AMREX_SPACEDIM>;

} // namespace

template <typename Normal>
void FitPlanesImpl (amrex::MultiFab& plane, amrex::MultiFab const& F, int ngrow)
{
    BL_PROFILE("PLIC::FitPlanes()");
    AMREX_ALWAYS_ASSERT(plane.nComp() == 4 && plane.nGrow() >= ngrow);
    AMREX_ALWAYS_ASSERT(F.nGrow() >= ngrow + Normal::stencil);

    plane.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(plane); mfi.isValid(); ++mfi) {
        const amrex::Box gbx = amrex::grow(mfi.validbox(), ngrow);
        auto const& Fa = F.const_array(mfi);
        auto const& pl = plane.array(mfi);
        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const amrex::Real f = Fa(i, j, k);
            if (f > pure_tol && f < 1.0_rt - pure_tol) {
                amrex::Real mx, my, mz, alpha;
                fit_plane<Normal>(i, j, k, Fa, mx, my, mz, alpha);
                pl(i, j, k, 0) = mx;
                pl(i, j, k, 1) = my;
                pl(i, j, k, 2) = mz;
                pl(i, j, k, 3) = alpha;
            }
        });
    }
    amrex::Gpu::streamSynchronize();
}

// Directions are processed in turn, so edge and corner ghosts see the ghosts of earlier directions.
//
// A ghost value depends on cells further along the wall than F has ghost cells, so that F alone
// would give different values in neighbouring boxes. The fill therefore works on a copy where the
// boxes extend through the walls: each wall ghost cell belongs to one box, its ghost layers cover
// the whole search, and they are exchanged between passes. The result does not depend on the box
// layout.
template <typename Normal>
void FillWallGhostsImpl (amrex::MultiFab& a_F, amrex::Geometry const& geom, ExtendFlags const& walls)
{
    BL_PROFILE("PLIC::FillWallGhosts()");
    const int nGhost = a_F.nGrow();
    AMREX_ALWAYS_ASSERT(nGhost >= Normal::stencil + 1);

    const auto is_wall = [&] (const int dir, const int sgn) {
        return !geom.isPeriodic(dir) && walls[dir][(sgn < 0) ? 0 : 1];
    };
    bool has_wall = false;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        has_wall = has_wall || is_wall(dir, -1) || is_wall(dir, 1);
    }
    if (!has_wall) {
        return;
    }

    // Passes beyond the first refine the ghost values at shallow contact angles, where the only
    // available planes read ghost cells
    constexpr int npass = 4;
    // The ghost layers read: the search along the wall, the reconstruction stencil of the plane it
    // finds, and the ghost layers of F themselves
    const int nwall = nGhost;
    constexpr int nread = wall_nsearch + Normal::stencil + 1;
    const amrex::Box& domain = geom.Domain();
    const auto periodicity = geom.periodicity();
    // The part of a box beyond a wall
    const auto beyond_wall = [&domain] (amrex::Box b, const int dir, const int sgn) {
        if (sgn < 0) {
            b.setBig(dir, domain.smallEnd(dir) - 1);
        } else {
            b.setSmall(dir, domain.bigEnd(dir) + 1);
        }
        return b;
    };

    // Boxes extended through the walls they touch. They stay disjoint.
    amrex::BoxList boxes;
    for (int ib = 0; ib < a_F.boxArray().size(); ++ib) {
        const amrex::Box b = a_F.boxArray()[ib];
        amrex::Box ext = b;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (is_wall(dir, -1) && b.smallEnd(dir) == domain.smallEnd(dir)) {
                ext.growLo(dir, nwall);
            }
            if (is_wall(dir, 1) && b.bigEnd(dir) == domain.bigEnd(dir)) {
                ext.growHi(dir, nwall);
            }
        }
        boxes.push_back(ext);
    }
    amrex::BoxArray ext_ba(std::move(boxes));
    amrex::MultiFab work(ext_ba, a_F.DistributionMap(), 1, nread);
    amrex::MultiFab snap(ext_ba, a_F.DistributionMap(), 1, nread);
    work.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(a_F); mfi.isValid(); ++mfi) {
        const amrex::Box bx = amrex::grow(mfi.validbox(), nGhost);
        work[mfi].template copy<amrex::RunOn::Device>(a_F[mfi], bx, 0, bx, 0, 1);
    }
    work.FillBoundary(periodicity);

    // The domain and its ghost layers, without bounds in periodic directions
    amrex::Box extent = domain;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        extent.grow(dir, geom.isPeriodic(dir) ? 1000000 : nwall);
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            if (!is_wall(dir, sgn)) {
                continue;
            }
            const int iwall = (sgn < 0) ? domain.smallEnd(dir) : domain.bigEnd(dir);

            // Cells of the wall-adjacent layer with a mixed cell of that layer within the search.
            // The passes of this wall do not change that layer. A ghost cell over any other cell
            // takes the value of the wall cell, without a search, and a wall with none needs no more
            // than one pass. The square of the search is dilated one direction at a time.
            amrex::LayoutData<amrex::IArrayBox> near(ext_ba, a_F.DistributionMap());
            int wall_contact = 0;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(max : wall_contact)
#endif
            for (amrex::MFIter mfi(work); mfi.isValid(); ++mfi) {
                if (!beyond_wall(mfi.validbox(), dir, sgn).ok()) {
                    continue;
                }
                amrex::Box scan = amrex::grow(mfi.validbox(), wall_nsearch);
                scan.setRange(dir, iwall);
                scan &= (work[mfi].box() & extent);
                near[mfi].resize(scan, 1);
                amrex::IArrayBox prev(scan, 1, amrex::The_Async_Arena());
                auto const& Fw = work.const_array(mfi);
                auto const& N = near[mfi].array();
                auto const& P = prev.array();
                amrex::ParallelFor(scan, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const amrex::Real f = Fw(i, j, k);
                    N(i, j, k) = static_cast<int>(f > pure_tol && f < 1.0_rt - pure_tol);
                });
                for (int t = 0; t < AMREX_SPACEDIM; ++t) {
                    if (t == dir) {
                        continue;
                    }
                    const amrex::IntVect et = amrex::IntVect::TheDimensionVector(t);
                    amrex::ParallelFor(scan, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        P(i, j, k) = N(i, j, k);
                    });
                    amrex::ParallelFor(scan, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        amrex::ignore_unused(k);
                        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
                        int any = 0;
                        for (int a = -wall_nsearch; a <= wall_nsearch && any == 0; ++a) {
                            const amrex::IntVect c = iv + a * et;
                            any = (scan.contains(c) && P(c) != 0) ? 1 : 0;
                        }
                        N(i, j, k) = any;
                    });
                }
                wall_contact = amrex::max(wall_contact, near[mfi].template max<amrex::RunOn::Device>());
            }
            amrex::Gpu::streamSynchronize();
            amrex::ParallelDescriptor::ReduceIntMax(wall_contact);

            // Passes beyond the first can read ghost values, the same way each time: once one
            // changes nothing, neither would the next.
            const int wall_npass = (wall_contact != 0) ? npass : 1;
            for (int pass = 0; pass < wall_npass; ++pass) {
                if (wall_contact != 0) {
                    amrex::MultiFab::Copy(snap, work, 0, 0, 1, nread);
                }
                const bool allow_ghost = (pass > 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(work); mfi.isValid(); ++mfi) {
                    const amrex::Box gbx = beyond_wall(mfi.validbox(), dir, sgn);
                    if (!gbx.ok()) {
                        continue;
                    }
                    const amrex::Box limit = work[mfi].box() & extent;
                    auto const& Fw = work.array(mfi);
                    auto const& S = snap.const_array(mfi);
                    auto const& N = near[mfi].const_array();

                    // One plane for each wall cell, shared by the ghost cells beyond it
                    amrex::Box wbx = mfi.validbox();
                    wbx.setRange(dir, iwall);
                    amrex::FArrayBox wall_plane(wbx, 8, amrex::The_Async_Arena());
                    auto const& PL = wall_plane.array();
                    if (wall_contact != 0) {
                        amrex::ParallelFor(wbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            amrex::ignore_unused(k);
                            const amrex::IntVect iv_w(AMREX_D_DECL(i, j, k));
                            amrex::IntVect ref = iv_w;
                            amrex::Real m[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                            amrex::Real alpha = 0.0_rt;
                            const bool found = (N(iv_w) != 0) &&
                                wall_contact_plane<Normal>(iv_w, dir, iwall, sgn, limit, S,
                                                           allow_ghost, ref, m, alpha);
                            PL(i, j, k, 0) = found ? 1.0_rt : 0.0_rt;
                            for (int n = 0; n < 3; ++n) {
                                PL(i, j, k, 1 + n) = m[n];
                            }
                            PL(i, j, k, 4) = alpha;
                            for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                                PL(i, j, k, 5 + n) = static_cast<amrex::Real>(ref[n]);
                            }
                        });
                    } else {
                        wall_plane.template setVal<amrex::RunOn::Device>(0.0_rt);
                    }

                    amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        amrex::ignore_unused(k);
                        const amrex::IntVect g(AMREX_D_DECL(i, j, k));
                        amrex::IntVect iv_w = g;
                        iv_w[dir] = iwall;
                        amrex::Real val = Fw(iv_w);
                        if (PL(iv_w, 0) > 0.5_rt) {
                            const amrex::Real m[3] = {PL(iv_w, 1), PL(iv_w, 2), PL(iv_w, 3)};
                            amrex::IntVect ref;
                            for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                                ref[n] = static_cast<int>(PL(iv_w, 5 + n));
                            }
                            val = wall_ghost_fraction(g, iv_w, ref, m, PL(iv_w, 4), Fw(iv_w));
                        }
                        Fw(g) = val;
                    });
                }
                amrex::Gpu::streamSynchronize();
                if (pass > 0) {
                    // The pass only changes this wall's ghost cells
                    auto const& work_ma = work.const_arrays();
                    auto const& snap_ma = snap.const_arrays();
                    int changed = amrex::ParReduce(
                        amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<int>{}, work,
                        amrex::IntVect(0),
                        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> amrex::GpuTuple<int>
                        {
                            return {static_cast<int>(work_ma[box_no](i, j, k) != snap_ma[box_no](i, j, k))};
                        });
                    amrex::ParallelDescriptor::ReduceIntMax(changed);
                    if (changed == 0) {
                        break;
                    }
                }
                work.FillBoundary(periodicity);
            }
        }
    }

    // Back to the ghost cells of F beyond the walls
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(a_F); mfi.isValid(); ++mfi) {
        const amrex::Box bx = amrex::grow(mfi.validbox(), nGhost);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                if (!is_wall(dir, sgn)) {
                    continue;
                }
                const amrex::Box gbx = beyond_wall(bx, dir, sgn);
                if (gbx.ok()) {
                    a_F[mfi].template copy<amrex::RunOn::Device>(work[mfi], gbx, 0, gbx, 0, 1);
                }
            }
        }
    }
    amrex::Gpu::streamSynchronize();
}

namespace {

[[noreturn]] void unknown_reconstruction (std::string const& reconstruction)
{
    amrex::Abort("Unknown reconstruction: " + reconstruction + " (expected youngs, myc or myc_height)");
    std::terminate();
}

// What the boundary condition of F on a domain face asks of FillWallGhosts: true to overwrite the
// ghost cells with the extended interface, false to keep the solver's values
bool extends_interface (int bc, int dir, int sgn)
{
    switch (bc) {
    case amrex::BCType::foextrap:
    case amrex::BCType::hoextrap:
    case amrex::BCType::hoextrapcc:
        return true;
    case amrex::BCType::int_dir:
    case amrex::BCType::ext_dir:
    case amrex::BCType::ext_dir_cc:
    case amrex::BCType::reflect_even:
    // Prescribed where the flow enters and extrapolated where it leaves, face by face: keep both, as
    // overwriting would also replace the volume fraction that flows in
    case amrex::BCType::direction_dependent:
        return false;
    default:
        amrex::Abort("PLIC::FillWallGhosts: boundary condition " + std::to_string(bc) + " on the "
                     + (sgn < 0 ? "low" : "high") + " side of direction " + std::to_string(dir)
                     + " is not valid for a volume fraction (expected int_dir, foextrap, hoextrap,"
                     + " ext_dir, reflect_even or direction_dependent)");
        return false;
    }
}

ExtendFlags extend_flags (amrex::BCRec const& bcrec)
{
    ExtendFlags extend{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        extend[dir][0] = extends_interface(bcrec.lo(dir), dir, -1);
        extend[dir][1] = extends_interface(bcrec.hi(dir), dir, 1);
    }
    return extend;
}

} // namespace

// One more than the stencil, for the first ghost layer of the planes. The advection schemes read no
// further than that.
int nGrowF (std::string const& reconstruction)
{
    if (reconstruction == "youngs") { return Youngs::stencil + 1; }
    if (reconstruction == "myc") { return MixedYoungsCentered::stencil + 1; }
    if (reconstruction == "myc_height") { return HeightFunctionMYC::stencil + 1; }
    unknown_reconstruction(reconstruction);
}

void FitPlanes (amrex::MultiFab& plane, amrex::MultiFab const& F, int ngrow,
                std::string const& reconstruction)
{
    if (reconstruction == "youngs") {
        FitPlanesImpl<Youngs>(plane, F, ngrow);
    } else if (reconstruction == "myc") {
        FitPlanesImpl<MixedYoungsCentered>(plane, F, ngrow);
    } else if (reconstruction == "myc_height") {
        FitPlanesImpl<HeightFunctionMYC>(plane, F, ngrow);
    } else {
        unknown_reconstruction(reconstruction);
    }
}

void FillWallGhosts (amrex::MultiFab& F, amrex::Geometry const& geom, amrex::BCRec const& bcrec,
                     std::string const& reconstruction)
{
    const ExtendFlags walls = extend_flags(bcrec);
    if (reconstruction == "youngs") {
        FillWallGhostsImpl<Youngs>(F, geom, walls);
    } else if (reconstruction == "myc") {
        FillWallGhostsImpl<MixedYoungsCentered>(F, geom, walls);
    } else if (reconstruction == "myc_height") {
        FillWallGhostsImpl<HeightFunctionMYC>(F, geom, walls);
    } else {
        unknown_reconstruction(reconstruction);
    }
}

} // namespace PLIC
/** @}*/
