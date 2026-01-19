"""Module defining lobster document schemas."""

import gzip
import logging
import time
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from emmet.core.structure import StructureMetadata
from monty.dev import requires
from monty.json import MontyDecoder
from monty.os.path import zpath
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.electronic_structure.cohp import Cohp
from pymatgen.io.lobster import Icohplist
from pymatgen.io.lobster.future import (
    CHARGE,
    CHARGE_LCFO,
    COBICAR,
    COBICAR_LCFO,
    COHPCAR,
    COHPCAR_LCFO,
    COOPCAR,
    DOSCAR,
    DOSCAR_LCFO,
    GROSSPOP,
    GROSSPOP_LCFO,
    ICOBILIST,
    ICOBILIST_LCFO,
    ICOHPLIST,
    ICOHPLIST_LCFO,
    ICOOPLIST,
    BandOverlaps,
    Fatbands,
    LobsterIn,
    LobsterOut,
    MadelungEnergies,
    NcICOBILIST,
    SitePotentials,
)
from typing_extensions import Self

from atomate2 import __version__

has_lobsterpy = bool(find_spec("lobsterpy"))
has_ijson = bool(find_spec("ijson"))

if TYPE_CHECKING or has_lobsterpy:
    from lobsterpy.cohp.analyze import Analysis
    from lobsterpy.cohp.describe import Description
    from pymatgen.io.lobster.future.core import LobsterFile

if TYPE_CHECKING or has_ijson:
    import ijson

logger = logging.getLogger(__name__)


class LobsteroutModel(BaseModel):
    """Definition of computational settings from the LOBSTER computation."""

    restart_from_projection: bool | None = Field(
        None,
        description="Bool indicating if the run has been restarted from a projection",
    )
    lobster_version: str | None = Field(None, description="Lobster version")
    threads: int | None = Field(
        None, description="Number of threads that Lobster ran on"
    )
    dft_program: str | None = Field(
        None, description="DFT program was used for this run"
    )
    charge_spilling: list[float] = Field(description="Absolute charge spilling")
    total_spilling: list[float] = Field(description="Total spilling")
    elements: list[str] = Field(description="Elements in structure")
    basis_type: list[str] = Field(description="Basis set used in Lobster")
    basis_functions: list[list[str]] = Field(description="basis_functions")
    timing: dict[str, dict[str, str]] = Field(description="Dict with infos on timing")
    warning_lines: list | None = Field(None, description="Warnings")
    info_orthonormalization: list | None = Field(
        None, description="additional information on orthonormalization"
    )
    info_lines: list | None = Field(
        None, description="list of strings with additional info lines"
    )
    has_doscar: bool | None = Field(
        None, description="Bool indicating if DOSCAR is present."
    )
    has_doscar_lso: bool | None = Field(
        None, description="Bool indicating if DOSCAR.LSO is present."
    )
    has_cohpcar: bool | None = Field(
        None, description="Bool indicating if COHPCAR is present."
    )
    has_coopcar: bool | None = Field(
        None, description="Bool indicating if COOPCAR is present."
    )
    has_cobicar: bool | None = Field(
        None, description="Bool indicating if COBICAR is present."
    )
    has_charge: bool | None = Field(
        None, description="Bool indicating if CHARGE is present."
    )
    has_madelung: bool | None = Field(
        None,
        description="Bool indicating if Site Potentials and Madelung file is present.",
    )
    has_projection: bool | None = Field(
        None, description="Bool indicating if projection file is present."
    )
    has_bandoverlaps: bool | None = Field(
        None, description="Bool indicating if BANDOVERLAPS file is present"
    )
    has_fatbands: bool | None = Field(
        None, description="Bool indicating if Fatbands are present."
    )
    has_grosspopulation: bool | None = Field(
        None, description="Bool indicating if GROSSPOP file is present."
    )
    has_density_of_energies: bool | None = Field(
        None, description="Bool indicating if DensityofEnergies is present"
    )


class LobsterinModel(BaseModel):
    """Definition of input settings for the LOBSTER computation."""

    cohpstartenergy: float = Field(description="Start energy for COHP computation")
    cohpendenergy: float = Field(description="End energy for COHP computation")

    gaussiansmearingwidth: float | None = Field(
        None, description="Set the smearing width in eV,default is 0.2 (eV)"
    )
    usedecimalplaces: int | None = Field(
        None,
        description="Set the decimal places to print in output files, default is 5",
    )
    cohpsteps: float | None = Field(
        None, description="Number steps in COHPCAR; similar to NEDOS of VASP"
    )
    basisset: str = Field(description="basis set of computation")
    cohpgenerator: str = Field(
        description="Build the list of atom pairs to be analyzed using given distance"
    )
    saveprojectiontofile: bool | None = Field(
        None, description="Save the results of projections"
    )
    lsodos: bool | None = Field(
        None, description="Writes DOS output from the orthonormalized LCAO basis"
    )
    basisfunctions: list[str] = Field(
        description="Specify the basis functions for element"
    )


class Bonding(BaseModel):
    """Model describing bonding field of BondsInfo."""

    integral: float | None = Field(
        None, description="Integral considering only bonding contributions from COHPs"
    )
    perc: float | None = Field(None, description="Percentage of bonding contribution")


class Antibonding(BaseModel):
    """Model describing antibonding field of BondsInfo."""

    integral: float | None = Field(
        None,
        description="Integral considering only anti-bonding contributions from COHPs",
    )
    perc: float | None = Field(
        None, description="Percentage of anti-bonding contribution"
    )


class BondsInfo(BaseModel):
    """Model describing bonds field of SiteInfo."""

    ICOHP_mean: str = Field(..., description="Mean of ICOHPs of relevant bonds")
    ICOHP_sum: str = Field(..., description="Sum of ICOHPs of relevant bonds")
    has_antibdg_states_below_Efermi: bool = Field(  # noqa: N815
        ...,
        description="Indicates if antibonding interactions below efermi are detected",
    )
    number_of_bonds: int = Field(
        ..., description="Number of bonds considered in the analysis"
    )
    bonding: Bonding = Field(description="Model describing bonding contributions")
    antibonding: Antibonding = Field(
        description="Model describing anti-bonding contributions"
    )


class SiteInfo(BaseModel):
    """Outer model describing sites field of Sites model."""

    env: str = Field(
        ...,
        description="The coordination environment identified from "
        "the LobsterPy analysis",
    )
    bonds: dict[str, BondsInfo] = Field(
        ...,
        description="A dictionary with keys as atomic-specie as key "
        "and BondsInfo model as values",
    )
    ion: str = Field(..., description="Ion to which the atom is bonded")
    charge: float = Field(..., description="Mulliken charge of the atom")
    relevant_bonds: list[str] = Field(
        ...,
        description="List of bond labels from the LOBSTER files i.e. for e.g. "
        " from ICOHPLIST.lobster/ COHPCAR.lobster",
    )


class Sites(BaseModel):
    """Model describing, sites field of CondensedBondingAnalysis."""

    sites: dict[int, SiteInfo] = Field(
        ...,
        description="A dictionary with site index as keys and SiteInfo model as values",
    )


class CohpPlotData(BaseModel):
    """Model describing the cohp_plot_data field of CondensedBondingAnalysis."""

    data: dict[str, Cohp] = Field(
        ...,
        description="A dictionary with plot labels from LobsterPy "
        "automatic analysis as keys and Cohp objects as values",
    )


class DictIons(BaseModel):
    """Model describing final_dict_ions field of CondensedBondingAnalysis."""

    data: dict[str, dict[str, int]] = Field(
        ...,
        description="Dict consisting information on environments of cations "
        "and counts for them",
    )


class DictBonds(BaseModel):
    """Model describing final_dict_bonds field of CondensedBondingAnalysis."""

    data: dict[str, dict[str, Union[float, bool]]] = Field(
        ..., description="Dict consisting information on ICOHPs per bond type"
    )


class CondensedBondingAnalysis(BaseModel):
    """Definition of condensed bonding analysis data from LobsterPy ICOHP."""

    formula: str = Field(description="Pretty formula of the structure")
    max_considered_bond_length: float = Field(
        description="Maximum bond length considered in bonding analysis"
    )
    limit_icohp: list[Union[str, float]] = Field(
        description="ICOHP range considered in co-ordination environment analysis"
    )
    number_of_considered_ions: int = Field(
        ..., description="number of ions detected based on Mulliken/Löwdin Charges"
    )
    sites: Sites = Field(
        ...,
        description="Bonding information at inequivalent sites in the structure",
    )
    type_charges: str = Field(
        description="Charge type considered for assigning valences in bonding analysis"
    )
    cutoff_icohp: float = Field(
        description="Percent limiting the ICOHP values to be considered"
        " relative to strongest ICOHP",
    )
    summed_spins: bool = Field(
        description="Bool that states if the spin channels in the "
        "cohp_plot_data are summed.",
    )
    start: float | None = Field(
        None,
        description="Sets the lower limit of energy relative to Fermi for evaluating"
        " bonding/anti-bonding percentages in the bond"
        " if set to None, all energies up-to the Fermi is considered",
    )
    cohp_plot_data: CohpPlotData = Field(
        ...,
        description="Plotting data for the relevant bonds from LobsterPy analysis",
    )
    which_bonds: str = Field(
        description="Specifies types of bond considered in LobsterPy analysis",
    )
    final_dict_bonds: DictBonds = Field(
        ...,
        description="Dict consisting information on ICOHPs per bond type",
    )
    final_dict_ions: DictIons = Field(
        ...,
        description="Model that describes final_dict_ions field",
    )
    run_time: float = Field(
        ..., description="Time needed to run Lobsterpy condensed bonding analysis"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Union[str, Path],
        save_cohp_plots: bool = True,
        lobsterpy_kwargs: dict | None = None,
        plot_kwargs: dict | None = None,
        which_bonds: str = "all",
    ) -> tuple:
        """Create a task document from a directory containing LOBSTER files.

        Parameters
        ----------
        dir_name : path or str
            The path to the folder containing the calculation outputs.
        save_cohp_plots : bool.
            Bool to indicate whether automatic cohp plots and jsons
            from lobsterpy will be generated.
        lobsterpy_kwargs : dict.
            kwargs to change default lobsterpy automatic analysis parameters.
        plot_kwargs : dict.
            kwargs to change plotting options in lobsterpy.
        which_bonds: str.
            mode for condensed bonding analysis: "cation-anion" and "all".
        """
        plot_kwargs = plot_kwargs or {}
        lobsterpy_kwargs = lobsterpy_kwargs or {}
        dir_name = Path(dir_name)
        cohpcar_path = Path(zpath(str((dir_name / "COHPCAR.lobster").as_posix())))
        charge_path = Path(zpath(str((dir_name / "CHARGE.lobster").as_posix())))
        structure_path = Path(zpath(str((dir_name / "CONTCAR").as_posix())))
        icohplist_path = Path(zpath(str((dir_name / "ICOHPLIST.lobster").as_posix())))
        icobilist_path = Path(zpath(str((dir_name / "ICOBILIST.lobster").as_posix())))
        icooplist_path = Path(zpath(str((dir_name / "ICOOPLIST.lobster").as_posix())))

        # Update lobsterpy analysis parameters with user supplied parameters
        lobsterpy_kwargs_updated = {
            "are_cobis": False,
            "are_coops": False,
            "cutoff_icohp": 0.10,
            "noise_cutoff": 0.1,
            "orbital_cutoff": 0.05,
            "orbital_resolved": False,
            "start": None,
            "summed_spins": False,  # we will always use spin polarization here
            "type_charge": None,
            **lobsterpy_kwargs,
        }

        try:
            start = time.time()
            analyse = Analysis(
                path_to_poscar=structure_path,
                path_to_icohplist=icohplist_path,
                path_to_cohpcar=cohpcar_path,
                path_to_charge=charge_path,
                which_bonds=which_bonds,
                **lobsterpy_kwargs_updated,
            )
            cba_run_time = time.time() - start
            # initialize lobsterpy condensed bonding analysis
            cba = analyse.condensed_bonding_analysis

            cba_cohp_plot_data = {}  # Initialize dict to store plot data

            seq_cohps = analyse.seq_cohps
            seq_labels_cohps = analyse.seq_labels_cohps
            seq_ineq_cations = analyse.seq_ineq_ions
            struct = analyse.structure

            for _iplot, (ication, labels, cohps) in enumerate(
                zip(seq_ineq_cations, seq_labels_cohps, seq_cohps, strict=True)
            ):
                label_str = f"{struct[ication].specie!s}{ication + 1!s}: "
                for label, cohp in zip(labels, cohps, strict=True):
                    if label is not None:
                        cba_cohp_plot_data[label_str + label] = cohp

            describe = Description(analysis_object=analyse)
            limit_icohp_val = list(cba["limit_icohp"])
            _replace_inf_values(limit_icohp_val)

            condensed_bonding_analysis = CondensedBondingAnalysis(
                formula=cba["formula"],
                max_considered_bond_length=cba["max_considered_bond_length"],
                limit_icohp=limit_icohp_val,
                number_of_considered_ions=cba["number_of_considered_ions"],
                sites=Sites(**cba),
                type_charges=analyse.type_charge,
                cohp_plot_data=CohpPlotData(data=cba_cohp_plot_data),
                cutoff_icohp=analyse.cutoff_icohp,
                summed_spins=lobsterpy_kwargs_updated.get("summed_spins"),
                which_bonds=analyse.which_bonds,
                final_dict_bonds=DictBonds(data=analyse.final_dict_bonds),
                final_dict_ions=DictIons(data=analyse.final_dict_ions),
                run_time=cba_run_time,
            )
            if save_cohp_plots:
                describe.plot_cohps(
                    save=True,
                    filename=f"automatic_cohp_plots_{which_bonds}.pdf",
                    hide=True,
                    **plot_kwargs,
                )
                import json

                filename = dir_name / f"condensed_bonding_analysis_{which_bonds}"
                with open(f"{filename}.json", "w") as fp:
                    json.dump(analyse.condensed_bonding_analysis, fp)
                with open(f"{filename}.txt", "w") as fp:
                    fp.writelines(f"{line}\n" for line in describe.text)

            # Read in strongest icohp values
            sb = _identify_strongest_bonds(
                analyse=analyse,
                icobilist_path=icobilist_path,
                icohplist_path=icohplist_path,
                icooplist_path=icooplist_path,
            )

        except ValueError:
            return None, None, None
        else:
            return condensed_bonding_analysis, describe, sb


class DosComparisons(BaseModel):
    """Model describing the DOS comparisons field in the CalcQualitySummary model."""

    tanimoto_orb_s: float | None = Field(
        None,
        description="Tanimoto similarity index between s orbital of "
        "VASP and LOBSTER DOS",
    )
    tanimoto_orb_p: float | None = Field(
        None,
        description="Tanimoto similarity index between p orbital of "
        "VASP and LOBSTER DOS",
    )
    tanimoto_orb_d: float | None = Field(
        None,
        description="Tanimoto similarity index between d orbital of "
        "VASP and LOBSTER DOS",
    )
    tanimoto_orb_f: float | None = Field(
        None,
        description="Tanimoto similarity index between f orbital of "
        "VASP and LOBSTER DOS",
    )
    tanimoto_summed: float | None = Field(
        None,
        description="Tanimoto similarity index for summed PDOS between "
        "VASP and LOBSTER",
    )
    e_range: list[Union[float, None]] = Field(
        description="Energy range used for evaluating the Tanimoto similarity index"
    )
    n_bins: int | None = Field(
        None,
        description="Number of bins used for discretizing the VASP and LOBSTER PDOS"
        "(Affects the Tanimoto similarity index)",
    )


class ChargeComparisons(BaseModel):
    """Model describing the charges field in the CalcQualitySummary model."""

    bva_mulliken_agree: bool | None = Field(
        None,
        description="Bool indicating whether atoms classification as cation "
        "or anion based on Mulliken charge signs of LOBSTER "
        "agree with BVA analysis",
    )
    bva_loewdin_agree: bool | None = Field(
        None,
        description="Bool indicating whether atoms classification as cations "
        "or anions based on Loewdin charge signs of LOBSTER "
        "agree with BVA analysis",
    )


class BandOverlapsComparisons(BaseModel):
    """Model describing the Band overlaps field in the CalcQualitySummary model."""

    file_exists: bool = Field(
        description="Boolean indicating whether the bandOverlaps.lobster "
        "file is generated during the LOBSTER run",
    )
    limit_maxDeviation: float | None = Field(  # noqa: N815
        None,
        description="Limit set for maximal deviation in pymatgen parser",
    )
    has_good_quality_maxDeviation: bool | None = Field(  # noqa: N815
        None,
        description="Boolean indicating whether the deviation at each k-point "
        "is within the threshold set using limit_maxDeviation "
        "for analyzing the bandOverlaps.lobster file data",
    )
    max_deviation: float | None = Field(
        None,
        description="Maximum deviation from ideal identity matrix from the observed in "
        "the bandOverlaps.lobster file",
    )
    percent_kpoints_abv_limit: float | None = Field(
        None,
        description="Percent of k-points that show deviations above "
        "the limit_maxDeviation threshold set in pymatgen parser.",
    )


class ChargeSpilling(BaseModel):
    """Model describing the Charge spilling field in the CalcQualitySummary model."""

    abs_charge_spilling: float = Field(
        description="Absolute charge spilling value from the LOBSTER calculation.",
    )
    abs_total_spilling: float = Field(
        description="Total charge spilling percent from the LOBSTER calculation.",
    )


class CalcQualitySummary(BaseModel):
    """Model describing the calculation quality of lobster run."""

    minimal_basis: bool = Field(
        description="Denotes whether the calculation used the minimal basis for the "
        "LOBSTER computation",
    )
    charge_spilling: ChargeSpilling = Field(
        description="Model describing the charge spilling from the LOBSTER runs",
    )
    charge_comparisons: ChargeComparisons | None = Field(
        None,
        description="Model describing the charge sign comparison results",
    )
    band_overlaps_analysis: BandOverlapsComparisons | None = Field(
        None,
        description="Model describing the band overlap file analysis results",
    )
    dos_comparisons: DosComparisons | None = Field(
        None,
        description="Model describing the VASP and LOBSTER PDOS comparisons results",
    )

    @classmethod
    @requires(
        has_lobsterpy, "lobsterpy must be installed to create an CalcQualitySummary."
    )
    def from_directory(
        cls,
        dir_name: Union[Path, str],
        calc_quality_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """Make a LOBSTER calculation quality summary from directory with LOBSTER files.

        Parameters
        ----------
        dir_name : path or str
            The path to the folder containing the calculation outputs.
        calc_quality_kwargs : dict
            kwargs to change calc quality analysis options in lobsterpy.

        Returns
        -------
        CalcQualitySummary
            A task document summarizing quality of the lobster calculation.
        """
        dir_name = Path(dir_name)
        calc_quality_kwargs = calc_quality_kwargs or {}
        band_overlaps_path = Path(
            zpath(str((dir_name / "bandOverlaps.lobster").as_posix()))
        )
        charge_path = Path(zpath(str((dir_name / "CHARGE.lobster").as_posix())))
        doscar_path = Path(
            zpath(str((dir_name / "DOSCAR.LSO.lobster").as_posix()))
            if Path(zpath(str((dir_name / "DOSCAR.LSO.lobster").as_posix()))).exists()
            else Path(zpath(str((dir_name / "DOSCAR.lobster").as_posix())))
        )
        lobsterin_path = Path(zpath(str((dir_name / "lobsterin").as_posix())))
        lobsterout_path = Path(zpath(str((dir_name / "lobsterout").as_posix())))
        potcar_path = (
            Path(zpath(str((dir_name / "POTCAR").as_posix())))
            if Path(zpath(str((dir_name / "POTCAR").as_posix()))).exists()
            else None
        )
        structure_path = Path(zpath(str((dir_name / "CONTCAR").as_posix())))
        vasprun_path = Path(zpath(str((dir_name / "vasprun.xml").as_posix())))

        # Update calc quality kwargs supplied by user
        calc_quality_kwargs_updated = {
            "e_range": [-20, 0],
            "dos_comparison": True,
            "n_bins": 256,
            "bva_comp": True,
            **calc_quality_kwargs,
        }

        cal_quality_dict = Analysis.get_lobster_calc_quality_summary(
            path_to_poscar=structure_path,
            path_to_vasprun=vasprun_path,
            path_to_charge=charge_path,
            path_to_potcar=potcar_path,
            path_to_doscar=doscar_path,
            path_to_lobsterin=lobsterin_path,
            path_to_lobsterout=lobsterout_path,
            path_to_bandoverlaps=band_overlaps_path,
            **calc_quality_kwargs_updated,
        )
        return CalcQualitySummary(**cal_quality_dict)


class StrongestBonds(BaseModel):
    """Strongest bonds extracted from ICOHPLIST/ICOOPLIST/ICOBILIST from LOBSTER.

    LobsterPy is used for the extraction.
    """

    which_bonds: str | None = Field(
        None,
        description="Denotes whether the information "
        "is for cation-anion pairs or all bonds",
    )
    strongest_bonds_icoop: dict | None = Field(
        None,
        description="Dict with infos on bond strength and bond length based on ICOOP.",
    )
    strongest_bonds_icohp: dict | None = Field(
        None,
        description="Dict with infos on bond strength and bond length based on ICOHP.",
    )
    strongest_bonds_icobi: dict | None = Field(
        None,
        description="Dict with infos on bond strength and bond length based on ICOBI.",
    )


class LobsterTaskDocument(StructureMetadata):
    """Definition of LOBSTER task document."""

    dir_name: str | Path = Field(description="The directory for this Lobster task")

    lobster_in: LobsterIn = Field(description="Lobster calculation inputs")
    lobster_out: LobsterOut | None = Field(None, description="Lobster out data")

    doscar: DOSCAR | None = Field(
        None, description="pymatgen pymatgen.io.lobster.Doscardata"
    )
    doscar_lcfo: DOSCAR_LCFO | None = Field(
        None, description="pymatgen pymatgen.io.lobster.DoscarLcfo data"
    )
    charge: CHARGE | None = Field(
        None,
        description="pymatgen Charge obj. Contains atomic charges based on Mulliken "
        "and Loewdin charge analysis",
    )
    charge_lcfo: CHARGE_LCFO | None = Field(
        None,
        description="pymatgen ChargeLcfo obj. Contains atomic charges based on "
        "Mulliken and Loewdin charge analysis for LCFO",
    )
    madelung_energies: MadelungEnergies | None = Field(
        None,
        description="pymatgen Madelung energies obj. Contains madelung energies"
        "based on Mulliken and Loewdin charges",
    )
    site_potentials: SitePotentials | None = Field(
        None,
        description="pymatgen Site potentials obj. Contains site potentials "
        "based on Mulliken and Loewdin charges",
    )
    gross_populations: GROSSPOP | None = Field(
        None,
        description="pymatgen Grosspopulations obj. Contains gross populations "
        " based on Mulliken and Loewdin charges ",
    )
    gross_populations_lcfo: GROSSPOP_LCFO | None = Field(
        None,
        description="pymatgen GrosspopulationsLcfo obj. Contains gross populations "
        " based on Mulliken and Loewdin charges for LCFO",
    )
    band_overlaps: BandOverlaps | None = Field(
        None,
        description="pymatgen Bandoverlaps obj for each k-point from"
        " bandOverlaps.lobster file if it exists",
    )

    cohpcar: COHPCAR | None = Field(
        None,
        description="pymatgen CompleteCohp object with COHP data",
    )
    coopcar: COOPCAR | None = Field(
        None, description="pymatgen CompleteCohp object with COOP data"
    )
    cobicar: COBICAR | None = Field(
        None, description="pymatgen CompleteCohp object with COBI data"
    )
    cohpcar_lcfo: COHPCAR_LCFO | None = Field(
        None,
        description="pymatgen CompleteCohp object with COHP data for LCFO",
    )
    cobicar_lcfo: COBICAR_LCFO | None = Field(
        None, description="pymatgen CompleteCohp object with COBI data for LCFO"
    )
    fatbands: Fatbands | None = Field(
        None, description="pymatgen Fatbands object with fatband data"
    )

    icohplist: ICOHPLIST | None = Field(
        None, description="pymatgen Icohplist object with ICOHP data"
    )
    icooplist: ICOOPLIST | None = Field(
        None, description="pymatgen Icohplist object with ICOOP data"
    )
    icobilist: ICOBILIST | None = Field(
        None, description="pymatgen Icohplist object with ICOBI data"
    )
    icobilist_lcfo: ICOBILIST_LCFO | None = Field(
        None, description="pymatgen Icohplist object with ICOBI data for LCFO"
    )
    icohplist_lcfo: ICOHPLIST_LCFO | None = Field(
        None, description="pymatgen Icohplist object with ICOHP data for LCFO"
    )
    nc_icobilist: NcICOBILIST | None = Field(
        None, description="pymatgen NcICOBILIST object with n-centers ICOBI data"
    )

    atomate2_version: str = Field(
        __version__, description="Version of atomate2 used to create the document"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        additional_fields: dict | None = None,
    ) -> Self:
        """Create a task document from a directory containing LOBSTER files.

        Parameters
        ----------
        dir_name : path or str.
            The path to the folder containing the calculation outputs.
        additional_fields : dict.
            Dictionary of additional fields to add to output document.

        Returns
        -------
        LobsterTaskDocument
            A task document for the lobster calculation.
        """
        logger = logging.getLogger(__name__)

        if isinstance(dir_name, str):
            dir_name = Path(dir_name)

        dir_name = dir_name.expanduser().resolve()

        additional_fields = additional_fields or {}

        lobster_objects: dict[str, type[LobsterFile]] = {
            "lobster_out": LobsterOut,
            "doscar": DOSCAR,
            "doscar_lcfo": DOSCAR_LCFO,
            "charge": CHARGE,
            "charge_lcfo": CHARGE_LCFO,
            "madelung_energies": MadelungEnergies,
            "site_potentials": SitePotentials,
            "gross_populations": GROSSPOP,
            "gross_populations_lcfo": GROSSPOP_LCFO,
            "band_overlaps": BandOverlaps,
            "cohpcar": COHPCAR,
            "coopcar": COOPCAR,
            "cobicar": COBICAR,
            "cohpcar_lcfo": COHPCAR_LCFO,
            "cobicar_lcfo": COBICAR_LCFO,
            "icohplist": ICOHPLIST,
            "icooplist": ICOOPLIST,
            "icobilist": ICOBILIST,
            "icobilist_lcfo": ICOBILIST_LCFO,
            "icohplist_lcfo": ICOHPLIST_LCFO,
            "nc_icobilist": NcICOBILIST,
        }

        lobster_kwargs: dict[str, Any] = dict.fromkeys(lobster_objects.keys(), None)

        for name, parser in lobster_objects.items():
            try:
                lobster_kwargs[name] = parser(
                    zpath(dir_name / parser.get_default_filename())
                )
            except FileNotFoundError:
                logger.warning(
                    f"Could not find the {name} file in {dir_name}", stacklevel=2
                )
            except RuntimeError:
                logger.exception(f"Could not parse {name} with {parser}")

        lobster_kwargs["lobster_in"] = LobsterIn.from_file(
            zpath(dir_name / "lobsterin")
        )

        meta_structure = Structure.from_file(zpath(dir_name / "CONTCAR"))

        try:
            lobster_kwargs["fatbands"] = Fatbands(dir_name)
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            logger.warning(f"Could not parse fatbands with Fatbands: {e}", stacklevel=2)
            lobster_kwargs["fatbands"] = None

        return cls.from_structure(
            meta_structure=meta_structure,
            dir_name=dir_name,
            **lobster_kwargs,
        ).model_copy(update=additional_fields)


def _replace_inf_values(data: Union[dict[Any, Any], list[Any]]) -> None:
    """
    Replace the -inf value in dictionary and with the string representation '-Infinity'.

    Parameters
    ----------
    data : dict
        dictionary to recursively iterate over

    Returns
    -------
    data
        Dictionary with replaced -inf values.

    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict | list):
                _replace_inf_values(
                    value
                )  # Recursively process nested dictionaries and lists
            elif value == float("-inf"):
                data[key] = "-Infinity"  # Replace -inf with a string representation
    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict | list):
                _replace_inf_values(
                    item
                )  # Recursively process nested dictionaries and lists
            elif item == float("-inf"):
                data[index] = "-Infinity"  # Replace -inf with a string representation


def _identify_strongest_bonds(
    analyse: Analysis,
    icobilist_path: Path,
    icohplist_path: Path,
    icooplist_path: Path,
) -> StrongestBonds:
    """
    Identify the strongest bonds and convert them into StrongestBonds objects.

    Parameters
    ----------
    analyse : .Analysis
        Analysis object from lobsterpy automatic analysis
    icobilist_path : Path or str
        Path to ICOBILIST.lobster
    icohplist_path : Path or str
        Path to ICOHPLIST.lobster
    icooplist_path : Path or str
        Path to ICOOPLIST.lobster

    Returns
    -------
    StrongestBonds
    """
    data = [
        (icohplist_path, False, False, "icohp"),
        (icobilist_path, True, False, "icobi"),
        (icooplist_path, False, True, "icoop"),
    ]
    output = []
    model_data = {"which_bonds": analyse.which_bonds}
    for file, are_cobis, are_coops, prop in data:
        if file.exists():
            icohplist = Icohplist(
                filename=file,
                are_cobis=are_cobis,
                are_coops=are_coops,
            )
            bond_dict = _get_strong_bonds(
                icohplist.icohpcollection.as_dict(),
                relevant_bonds=analyse.final_dict_bonds,
                are_cobis=are_cobis,
                are_coops=are_coops,
            )
            model_data[f"strongest_bonds_{prop}"] = bond_dict
            output.append(
                StrongestBonds(
                    strongest_bonds=bond_dict,
                    which_bonds=analyse.which_bonds,
                )
            )
        else:
            model_data[f"strongest_bonds_{prop}"] = {}
            output.append(None)
    return StrongestBonds(**model_data)


# Don't we have this in pymatgen somewhere?
def _get_strong_bonds(
    bondlist: dict, are_cobis: bool, are_coops: bool, relevant_bonds: dict
) -> dict:
    """
    Identify the strongest bonds from a list of bonds.

    Parameters
    ----------
    bondlist : dict.
        dict including bonding information
    are_cobis : bool.
        True if these are cobis
    are_coops : bool.
        True if these are coops
    relevant_bonds : dict.
        Dict include all bonds that are considered.

    Returns
    -------
    dict
        Dictionary including the strongest bonds.
    """
    bonds = []
    icohp_all = []
    lengths = []
    for a, b, c, length in zip(
        bondlist["list_atom1"],
        bondlist["list_atom2"],
        bondlist["list_icohp"],
        bondlist["list_length"],
        strict=True,
    ):
        bonds.append(f"{a.rstrip('0123456789')}-{b.rstrip('0123456789')}")
        icohp_all.append(sum(c.values()))
        lengths.append(length)

    bond_labels_unique = list(set(bonds))
    sep_icohp: list[list[float]] = [[] for _ in range(len(bond_labels_unique))]
    sep_lengths: list[list[float]] = [[] for _ in range(len(bond_labels_unique))]

    for idx, val in enumerate(bond_labels_unique):
        for j, val2 in enumerate(bonds):
            if val == val2:
                sep_icohp[idx].append(icohp_all[j])
                sep_lengths[idx].append(lengths[j])

    if are_cobis and not are_coops:
        prop = "icobi"
    elif not are_cobis and are_coops:
        prop = "icoop"
    else:
        prop = "icohp"

    bond_dict: dict[str, dict[str, Union[float, str]]] = {}
    for idx, lab in enumerate(bond_labels_unique):
        label = lab.split("-")
        label.sort()
        for rel_bnd in relevant_bonds:
            rel_bnd_list = rel_bnd.split("-")
            rel_bnd_list.sort()
            if label == rel_bnd_list:
                if prop == "icohp":
                    index = np.argmin(sep_icohp[idx])
                    bond_dict |= {
                        rel_bnd: {
                            "bond_strength": min(sep_icohp[idx]),
                            "length": sep_lengths[idx][index],
                        }
                    }
                else:
                    index = np.argmax(sep_icohp[idx])
                    bond_dict |= {
                        rel_bnd: {
                            "bond_strength": max(sep_icohp[idx]),
                            "length": sep_lengths[idx][index],
                        }
                    }
    return bond_dict


@requires(has_ijson, "ijson must be installed to read lobster json files.")
def read_saved_json(
    filename: str, pymatgen_objs: bool = True, query: str = "structure"
) -> dict[str, Any]:
    r"""
    Read the data from  \*.json.gz files corresponding to query.

    Uses ijson to parse specific keys(memory efficient)

    Parameters
    ----------
    filename: str.
        name of the json file to read
    pymatgen_objs: bool.
        if True will convert structure,coop, cobi, cohp and dos to pymatgen objects
    query: str or None.
        field name to query from the json file. If None, all data will be returned.

    Returns
    -------
    dict
        Returns a dictionary with lobster task json data corresponding to query.
    """
    with gzip.open(filename, "rb") as file:
        lobster_data = {
            field: data
            for obj in ijson.items(file, "item", use_float=True)
            for field, data in obj.items()
            if query is None or query in obj
        }
        if not lobster_data:
            raise ValueError(
                "Please recheck the query argument. "
                f"No data associated to the requested 'query={query}' "
                f"found in the JSON file"
            )
    if pymatgen_objs:
        for query_key, value in lobster_data.items():
            if isinstance(value, dict):
                lobster_data[query_key] = MontyDecoder().process_decoded(value)
            elif "lobsterpy_data" in query_key:
                for field in lobster_data[query_key].__fields__:
                    val = MontyDecoder().process_decoded(
                        getattr(lobster_data[query_key], field)
                    )
                    setattr(lobster_data[query_key], field, val)

    return lobster_data
