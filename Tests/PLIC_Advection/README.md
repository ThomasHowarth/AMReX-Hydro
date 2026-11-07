# PLIC_Advection

Verification of the PLIC volume-of-fluid library (`PLIC`), as a small solver would use it: an
`AmrCore` that owns the mesh and the ghost cells, with a prescribed velocity. `PLICDrivers.H` advances
the volume fraction with the operators of the library, on all levels with the same time step. A run
exits with a non-zero status when one of its checks does not hold.

```
make -j                  # DIM = 2 by default; DIM=3 for inputs.3d.deform
mpirun -np 4 ./main2d.gnu.MPI.ex inputs.kernels
mpirun -np 4 ./main2d.gnu.MPI.ex inputs.2d.vortex plic.advection=<scheme>
```

| Input                | Case                                                            |
|----------------------|-----------------------------------------------------------------|
| `inputs.kernels`     | Plane geometry, normals, wall ghost cells, face density         |
| `inputs.2d.vortex`   | Circle in the time-reversed single vortex, 3 levels             |
| `inputs.2d.zalesak`  | Zalesak's slotted disk, one revolution, 3 levels                |
| `inputs.2d.wall_vortex` | Circle cut by a wall, in the same flow: a free contact line  |
| `inputs.3d.deform`   | Sphere in LeVeque's deformation field, 3 levels                 |

The advection cases return to their initial shape at `t = prob.period`, so the shape error is the L1
difference from the initial volume fraction. `plic.advection` selects the scheme of
the driver, and `plic.reconstruction` the reconstruction (default `myc_height`). Set
`amr.max_level = 0` and `amr.n_cell` for a uniform grid. The `test.max_*` parameters set the checks.

## Expected results

`inputs.kernels` checks that a plane holds the volume of its cell, that the `myc_height` normal is
exact for a plane, that wall ghost cells continue a planar interface, and that the face density
balances gravity normal to a plane. Each check is to round-off.
