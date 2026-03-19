"""Utility functions for working with LOBSTER workflow."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymatgen.core import IMolecule, IStructure

from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer


def get_connectivity_matrix(
    structure: IStructure | IMolecule, cutoff_multiplier: float = np.sqrt(2)
) -> NDArray[np.integer]:
    """
    Get the connectivity matrix of a structure or molecule based on covalent radii.

    Parameters
    ----------
    structure : Structure or Molecule
        The structure or molecule for which to compute the connectivity matrix.
    cutoff_multiplier : float, optional
        A multiplier to adjust the covalent radii when determining connectivity.

    Returns
    -------
    NDArray[np.integer]
        A connectivity matrix where element (i, j) is 1 if atoms i and j are connected,
        and 0 otherwise.
    """
    covalent_radius = (
        np.array([CovalentRadius.radius[site.specie.symbol] for site in structure])
        * cutoff_multiplier
    )

    cutoffs = np.add.outer(covalent_radius, covalent_radius)

    connectivity_matrix = (structure.distance_matrix < cutoffs).astype(int)
    np.fill_diagonal(connectivity_matrix, 0)

    return connectivity_matrix


def get_cell_neighbors(
    structure: IStructure, indices: NDArray[np.integer]
) -> NDArray[np.integer]:
    """
    Return the cell indices of neighbors for given atom indices.

    Parameters
    ----------
    structure : Structure | Molecule
        The structure or molecule containing the atoms.
    indices : NDArray[np.integer]
        A 2D array where each row contains the indices of neighboring atoms.

    Returns
    -------
    NDArray[np.integer]
        A 2D array where each row contains the cell indices of the neighbors
        corresponding to the input indices.
    """
    scaled_positions = structure.frac_coords

    reference_positions = scaled_positions[indices[:, 0], :][:, None, :]
    scaled_positions = scaled_positions[indices, :]

    scaled_distances = scaled_positions - reference_positions

    cells = np.zeros_like(scaled_distances, dtype=int)
    cells[scaled_distances > 0.5] = -1
    cells[scaled_distances < -0.5] = 1

    return cells


def get_molecule_unique_sites(
    site_collection: IMolecule,
) -> NDArray[np.integer]:
    """
    Get unique sites from a pymatgen Molecule object.

    Parameters
    ----------
    site_collection : IMolecule
        The molecule for which to get unique sites.

    Returns
    -------
    NDArray[np.integer]
        An array of unique site indices.
    """
    point_group_analyzer = PointGroupAnalyzer(site_collection)

    return np.array(list(point_group_analyzer.get_equivalent_atoms()["eq_sets"].keys()))


def get_structure_unique_sites(
    site_collection: IStructure,
) -> NDArray[np.integer]:
    """
    Get unique sites from a pymatgen Structure object.

    Parameters
    ----------
    site_collection : IStructure
        The structure for which to get unique sites.

    Returns
    -------
    NDArray[np.integer]
        An array of unique site indices.
    """
    spacegroup_analyzer = SpacegroupAnalyzer(site_collection)

    equivalent_indices = (
        spacegroup_analyzer.get_symmetrized_structure().equivalent_indices
    )

    return np.array([eq[0] for eq in equivalent_indices])


def get_lobster_three_centers_cobibetween_input_dict(
    multi_center_bonds: NDArray[np.integer],
    cell_indices: NDArray[np.integer],
    orbital_wise: bool = False,
) -> dict[str, list[str]]:
    """
    Create a dictionary for Lobster input based on multi-center bonds and cell indices.

    Parameters
    ----------
    multi_center_bonds : list[tuple[int, int]]
        A list of tuples representing the indices of atoms involved in multi-center
        bonds.
    cell_indices : list[int]
        A list of indices representing the cells in which the atoms are located,
        required by Lobster.
    orbital_wise : bool, optional
        Whether to include the 'orbitalwise' option in the Lobster input, making the
        calculation orbital-wise.

    Returns
    -------
    dict
        A dictionary suitable for Lobster input.
    """
    return {
        "cobiBetween": [
            f"atom1 {i} atom2 {j} cell {'{} {} {}'.format(*b)} atom3 {k} cell "
            f"{'{} {} {}'.format(*c)} {'orbitalwise' if orbital_wise else ''}"
            for (i, j, k), (_, b, c) in zip(
                multi_center_bonds, cell_indices, strict=True
            )
        ]
    }


def get_three_center_bonds(
    connectivity_matrix: NDArray[np.integer],
    unique_sites: NDArray[np.integer] | list[int],
) -> NDArray[np.integer]:
    """
    Get potential three-center bonds for a given connectivity matrix and unique sites.

    This function identifies pairs of atoms that are connected to a
    third atom, indicating potential three-center bonds.

    Parameters
    ----------
    connectivity_matrix : NDArray[np.integer]
        A 2D array representing the connectivity matrix of the structure.
    unique_sites : NDArray[np.integer]
        A 1D array of indices representing the unique sites of the atoms in the
        structure.

    Returns
    -------
    NDArray[np.integer]
        A 2D array where each row represents a multi-center bond in the form
        [atom1_index, atom2_index, central_atom_index]. The indices correspond to the
        unique sites of the atoms in the structure.
    """
    reduced_connectivity = connectivity_matrix[unique_sites, :]
    indices, neighbors = np.where(reduced_connectivity == 1)

    all_mcbs = []

    for k in np.unique(indices):
        all_neighbors = neighbors[indices == k]

        possible_mcbs = list(combinations(all_neighbors, 2))
        possible_mcbs = np.column_stack(
            (possible_mcbs, np.full(len(possible_mcbs), unique_sites[k]))
        ).astype(int)

        if len(possible_mcbs) > 0:
            all_mcbs.append(np.sort(possible_mcbs))

    return np.unique(np.vstack(all_mcbs), axis=0)
