"""A module defining flows for LOBSTER calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jobflow.core.maker import Maker

from atomate2.lobster.jobs import LobsterMaker
from atomate2.lobster.utils import (
    get_cell_neighbors,
    get_connectivity_matrix,
    get_lobster_three_centers_cobibetween_input_dict,
    get_three_center_bonds,
)
from atomate2.vasp.flows.lobster import VaspLobsterMaker

if TYPE_CHECKING:
    from jobflow.core.flow import Flow
    from numpy.typing import NDArray
    from pymatgen.core import IStructure

    from atomate2.vasp.flows.core import UniformBandStructureMaker
    from atomate2.vasp.jobs.core import RelaxMaker


@dataclass
class MCBAMaker(Maker):
    """
    Maker to create a flow for performing multi-center bond analysis using Lobster.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    orbital_wise : bool
        Whether to perform orbital-wise multi-center bond analysis.
    molecule_box_size : float
        The size of the box to create around molecules (in Angstroms) if the input is a
        molecule.
    additional_fields : dict[str, Any]
        Additional fields to include in the flow metadata.
    """

    name: str = "MCBALobsterFlow"  # type: ignore[assignment]
    vasp_relax_maker: RelaxMaker | None = None
    vasp_static_maker: UniformBandStructureMaker | None = None
    lobster_maker: LobsterMaker | None = None
    orbital_wise: bool = False

    def make(
        self,
        site_collection: IStructure,
        connectivity_matrix: NDArray[np.integer] | None = None,
        unique_sites: NDArray[np.integer] | None = None,
    ) -> Flow:
        """
        Create a flow for performing a multi-center bond Lobster calculation.

        This method attempts to identify multi-center bonds in the provided structure
        using cutoff and sets up a Lobster calculation accordingly.

        Parameters
        ----------
        structures : Structure
            The structure for which to create the flow.
        vasp_relax_maker : RelaxMaker | None, optional
            A `RelaxMaker` for the VASP relaxation step. If None, no relaxation will be
            performed.
        vasp_static_maker : UniformBandStructureMaker | None, optional
            A `UniformBandStructureMaker` for the VASP static step. If None, a default
            will be used.
        lobster_maker : LobsterMaker | None, optional
            A `LobsterMaker` for the Lobster calculation. If None, a default will be
            used.

        Returns
        -------
        Flow
            A Flow object containing the LobsterMaker jobs for each structure.
        """
        unique_sites = unique_sites or np.arange(len(site_collection))

        connectivity_matrix = connectivity_matrix or get_connectivity_matrix(
            site_collection, cutoff_multiplier=np.sqrt(2)
        )

        multi_center_bonds = get_three_center_bonds(
            connectivity_matrix=connectivity_matrix,
            unique_sites=unique_sites,
        )

        if len(multi_center_bonds) == 0:
            raise ValueError(
                "No multi-center bonds found in the provided structure with the given "
                "settings."
            )

        cell_indices = get_cell_neighbors(site_collection, multi_center_bonds)

        lobster_input_dict = get_lobster_three_centers_cobibetween_input_dict(
            multi_center_bonds=multi_center_bonds,
            cell_indices=cell_indices,
            orbital_wise=self.orbital_wise,
        )

        lobster_maker = self.lobster_maker or LobsterMaker(
            calculation_type="standard_with_energy_range_from_vasprun",
        )
        user_settings = lobster_maker.user_lobsterin_settings or {}

        lobster_maker.user_lobsterin_settings = {**user_settings, **lobster_input_dict}

        vasp_lobster_maker = VaspLobsterMaker(
            relax_maker=self.vasp_relax_maker,
            lobster_static_maker=self.vasp_static_maker,  # type: ignore[arg-type]
            lobster_maker=lobster_maker,  # type: ignore[arg-type]
        )

        return vasp_lobster_maker.make(site_collection)
