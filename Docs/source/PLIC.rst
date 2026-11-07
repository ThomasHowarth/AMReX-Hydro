.. include:: CustomCommands.rst

.. _plic:

Volume of fluid with PLIC
=========================

The ``PLIC`` directory provides geometric volume-of-fluid transport with a piecewise-linear
interface calculation, for any AMReX-based solver. The interface in a cell is the plane
:math:`\mathbf{m} \cdot \mathbf{x} = \alpha` in unit-cube cell coordinates, with the liquid on the
side :math:`\mathbf{m} \cdot \mathbf{x} \le \alpha`, and the plane is fixed by the volume fraction
:math:`F` of the cell through the analytic relations of Scardovelli & Zaleski (JCP 164, 2000).

The library works on the MultiFabs of the solver and holds no solver state. It requires Cartesian
coordinates and does not support embedded boundaries. Like the rest of AMReX-Hydro it provides spatial
operators on one level, and leaves the time integration to the solver: the update of :math:`F`, its
ghost cells, the synchronization of the levels and any subcycling. The reconstruction is selected by
name, as ``advection_type`` is in ``HydroUtils::ComputeFluxesOnBoxFromState``, and each advection
scheme has its own namespace.

Fluxes
------

Two conventions differ from ``HydroUtils::ComputeFluxesOnBoxFromState``, because the fluxes are
geometric:

- A flux is the liquid *volume* that crosses a face over the step, per unit face area, rather than a
  rate :math:`u s`. The update is :math:`F \leftarrow F + \sum_d (\phi_{lo} - \phi_{hi}) / \Delta x_d`,
  with no factor of :math:`\Delta t`. For refluxing with subcycling, the time factor of the flux register
  is therefore 1, not :math:`\Delta t`, and the area factor is the face area.
- The operators work on all the boxes of one level, rather than one box: a mesh node shared by several
  boxes must be traced back to one position, so that the flux volumes tile, and the wall ghost cells
  depend on cells further along the wall than one box holds.

What the solver provides
------------------------

The ghost cells of :math:`F` depend on the boundary conditions of the solver. They should

- use ``amrex::cell_cons_interp`` at coarse/fine boundaries, which keeps :math:`F` in :math:`[0,1]`;
- be filled with the ``BCRec`` of :math:`F`, as for any other scalar, and then passed with that
  ``BCRec`` to ``PLIC::FillWallGhosts``.

``PLIC::FillWallGhosts`` reads the boundary condition of each domain face:

.. list-table::
   :header-rows: 1

   * - ``BCType``
     - Boundary
     - Ghost cells
   * - ``int_dir``
     - periodic
     - left alone
   * - ``foextrap``, ``hoextrap``
     - wall or outflow
     - overwritten: the planes of the interior interface are extended through the face
   * - ``ext_dir``
     - inflow
     - kept, as the volume fraction that flows in
   * - ``reflect_even``
     - symmetry
     - kept, as the mirror image: the interface meets the plane at 90 degrees
   * - ``direction_dependent``
     - inflow or outflow, face by face
     - kept
   * - ``reflect_odd`` or any other
     - --
     - aborts

At a wall the extension gives a free contact line: the interface meets the wall at whatever angle it
has, while extrapolated values would impose 90 degrees. At an outflow it continues the interface
smoothly. The result does not depend on the box layout. Near the edges of a domain, where the
stencils read the ghost cells of another face, it is only approximate for a planar interface.

The face velocities must have one filled ghost layer, and be at :math:`t^{n+1/2}` and discretely
divergence-free. The solver also owns the projection, for which ``PLIC::SharpFaceDensity`` gives a
face density that balances gravity for a planar interface (Francois et al., JCP 213, 2006). The
helpers ``PLIC::FillMixture``, ``PLIC::EstimateDt``, ``PLIC::LiquidVolume`` and
``PLIC::TagInterface`` cover mixture properties, the time step, diagnostics and refinement.

.. note::

   Keep the interface on the finest level, with ``PLIC::TagInterface``: flux volumes do not tile
   across coarse/fine boundaries, which must stay in pure cells. With subcycling, the reflux
   corrections then fall in pure cells too.

Reconstructions
---------------

=============== ===================================================================================
``youngs``      Youngs' finite-difference normal.
``myc``         Mixed Youngs-Centered normal (Aulisa et al., JCP 225, 2007).
``myc_height``  The height-function normal along the dominant direction of the Mixed
                Youngs-Centered normal, which is exact for a planar interface, and the latter where
                the heights are not defined. The default.
=============== ===================================================================================

Advection schemes
-----------------

Each scheme is added in its own change.

Adding a scheme
---------------

Schemes are added to the library, in the same way as for Godunov, MOL and BDS.

**A new interface normal** runs inside device kernels, so it is a template parameter rather than a
virtual function. Write a struct in ``hydro_plic_normals_K.H``, as described there, and add its name
to ``nGrowF``, ``FitPlanes`` and ``FillWallGhosts`` in ``hydro_plic_reconstruction.cpp``. Plane fitting,
wall ghost cells and all the advection schemes then work with it.

**A new advection scheme** goes in its own namespace and files, with the operators on one level that a
solver needs for it: its fluxes, from ``PLIC::FitPlanes``, and any update that is not a plain flux
difference. A driver for it in ``Tests/PLIC_Advection/PLICDrivers.H`` makes it available to the tests.

Verification
------------

``Tests/PLIC_Advection`` is a small AMR solver over the library, with a prescribed velocity from
``PLIC::FaceVelocityFromPotential``, and inputs for Zalesak's disk, the single vortex, the 3D
deformation field and a contact line. A run fails when its checks do not hold. See its README for
the expected results.
