"""A module defining flows for LOBSTER calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow.core.maker import Maker

from atomate2.lobster.jobs import LobsterMaker
from atomate2.lobster.utils import (
    get_cell_neighbors,
    get_connectivity_matrix,
    get_lobster_three_centers_cobibetween_input_dict,
    get_structure_unique_sites,
    get_three_center_bonds,
)
from atomate2.vasp.flows.lobster import LOBSTER_UNIFORM_MAKER, VaspLobsterMaker

if TYPE_CHECKING:
    from jobflow.core.flow import Flow
    from numpy.typing import NDArray
    from pymatgen.core import Structure

    from atomate2.vasp.flows.core import UniformBandStructureMaker
    from atomate2.vasp.jobs.core import BaseVaspMaker, RelaxMaker


@dataclass
class LobsterMCBMaker(Maker):
    """
    Maker to create a flow for performing multi-center bond analysis using LOBSTER.

    Constructs a jobflow ``Flow`` that runs VASP (optional relaxation + static) followed
    by a LOBSTER calculation configured with COBIBETWEEN keywords derived from the
    three-center bonds found in the structure.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker. Default is ``"LobsterMCBMaker"``.
    vasp_relax_maker : RelaxMaker or None
        Maker used for the VASP structure relaxation step. If ``None``, the default
        ``RelaxMaker`` configured inside ``VaspLobsterMaker`` is used.
    vasp_static_maker : UniformBandStructureMaker or None
        Maker used for the VASP static (uniform band structure) calculation. If
        ``None``, the default ``UniformBandStructureMaker`` inside ``VaspLobsterMaker``
        is used.
    lobster_maker : LobsterMaker or None
        Maker used to run the LOBSTER calculation. If ``None``, a ``LobsterMaker``
        with ``calculation_type="standard_with_energy_range_from_vasprun"`` is
        created automatically.
    orbital_wise : bool
        Whether to request orbital-resolved COBIBETWEEN output from LOBSTER.
        Default is ``False``.
    """

    name: str = "LobsterMCBMaker"  # type: ignore[assignment]
    vasp_relax_maker: RelaxMaker | None = field(default=None)
    vasp_static_maker: UniformBandStructureMaker | BaseVaspMaker | None = field(
        default_factory=lambda: LOBSTER_UNIFORM_MAKER
    )
    lobster_maker: LobsterMaker = field(
        default_factory=lambda: LobsterMaker(
            calculation_type="standard_with_energy_range_from_vasprun",
        )
    )
    orbital_wise: bool = False

    def make(
        self,
        structure: Structure,
        multi_center_bonds: NDArray[np.integer] | None = None,
        use_symmetry_detection: bool = True,
    ) -> Flow:
        """
        Create a flow for performing a multi-center bond LOBSTER calculation.

        If ``multi_center_bonds`` is not provided, three-center bonds are detected
        automatically from pairwise connectivity derived from scaled covalent radii.
        The detected (or supplied) bonds are translated into LOBSTER COBIBETWEEN
        keywords and merged with any user-supplied LOBSTER settings before the flow
        is assembled.

        Parameters
        ----------
        structure : Structure
            The input structure for which the flow will be created.
        multi_center_bonds : NDArray[np.integer] of shape (N, 3) or None
            Integer indices describing the N three-center bonds, where each row
            contains the indices of the three atoms forming one bond. If ``None``,
            bonds are detected automatically using a connectivity matrix built with a
            cutoff multiplier of ``sqrt(2)`` times the sum of covalent radii.

        Returns
        -------
        Flow
            A jobflow ``Flow`` containing the VASP and LOBSTER jobs required to
            evaluate multi-center bond orders for the given structure.

        Raises
        ------
        ValueError
            If no three-center bonds are provided and none can be detected in the
            structure.
        """
        if multi_center_bonds is None:
            unique_sites = (
                get_structure_unique_sites(structure)
                if use_symmetry_detection
                else np.arange(len(structure))
            )

            connectivity_matrix = get_connectivity_matrix(
                structure, cutoff_multiplier=np.sqrt(2)
            )

            multi_center_bonds = get_three_center_bonds(
                connectivity_matrix=connectivity_matrix,
                unique_sites=unique_sites,
            )

        if len(multi_center_bonds) == 0:
            raise ValueError(
                "No multi-center bonds provided or found in the given structure."
            )

        cell_indices = get_cell_neighbors(structure, multi_center_bonds)

        lobster_input_dict = get_lobster_three_centers_cobibetween_input_dict(
            multi_center_bonds=multi_center_bonds,
            cell_indices=cell_indices,
            orbital_wise=self.orbital_wise,
        )

        # lobster_maker = self.lobster_maker or LobsterMaker(
        #    calculation_type="standard_with_energy_range_from_vasprun",
        # )
        user_settings = self.lobster_maker.user_lobsterin_settings or {}

        self.lobster_maker.user_lobsterin_settings = {
            **user_settings,
            **lobster_input_dict,
        }

        vasp_lobster_maker = VaspLobsterMaker(
            relax_maker=self.vasp_relax_maker,
            lobster_static_maker=self.vasp_static_maker,  # type: ignore[arg-type]
            lobster_maker=self.lobster_maker,  # type: ignore[arg-type]
        )

        return vasp_lobster_maker.make(structure)
