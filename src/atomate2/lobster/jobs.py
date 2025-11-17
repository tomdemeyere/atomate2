"""Module defining lobster jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from jobflow.core.job import job
from jobflow.core.maker import Maker
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.dos import LobsterCompleteDos
from pymatgen.io.lobster import Bandoverlaps, Icohplist, Lobsterin

from atomate2 import SETTINGS
from atomate2.common.files import gzip_output_folder
from atomate2.lobster.files import (
    LOBSTEROUTPUT_FILES,
    VASP_OUTPUT_FILES,
    copy_lobster_files,
)
from atomate2.lobster.future.schemas import LobsterTaskDocument
from atomate2.lobster.run import run_lobster

logger = logging.getLogger(__name__)


_FILES_TO_ZIP = [*LOBSTEROUTPUT_FILES, "lobsterin", *VASP_OUTPUT_FILES]


@dataclass
class LobsterMaker(Maker):
    """
    LOBSTER job maker.

    The maker copies DFT output files necessary for the LOBSTER run. It will create all
    lobsterin files, run LOBSTER, zip the outputs and parse the LOBSTER outputs.

    Parameters
    ----------
    name : str
        Name of jobs produced by this maker.
    task_document_kwargs : dict
        Keyword arguments passed to :obj:`.LobsterTaskDocument.from_directory`.
    user_lobsterin_settings : dict
        Dict including additional information on the Lobster settings.
    run_lobster_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_lobster`.
    calculation_type : str
        Type of calculation for the Lobster run that will get passed to
        :obj:`.Lobsterin.standard_calculations_from_vasp_files`.
    """

    name: str = "lobster"
    additional_fields: dict = field(default_factory=dict)
    user_lobsterin_settings: dict | None = None
    run_lobster_kwargs: dict = field(default_factory=dict)
    calculation_type: str = "standard"

    @job(output_schema=LobsterTaskDocument)
    def make(
        self,
        wavefunction_dir: str | Path | None = None,
        basis_dict: dict | None = None,
    ) -> LobsterTaskDocument:
        """Run a LOBSTER calculation.

        Parameters
        ----------
        wavefunction_dir : str | Path | None
            A directory containing a WAVEFUNCTION and other outputs needed for Lobster
        basis_dict: dict
            A dict including information on the basis set
        """
        copy_lobster_files(wavefunction_dir)

        lobsterin = Lobsterin.standard_calculations_from_vasp_files(
            "POSCAR", "INCAR", dict_for_basis=basis_dict, option=self.calculation_type
        )

        if self.user_lobsterin_settings:
            for key, parameter in self.user_lobsterin_settings.items():
                if key != "basisfunctions":
                    lobsterin[key] = parameter

        lobsterin.write_lobsterin("lobsterin")

        logger.info("Running LOBSTER")
        run_lobster(**self.run_lobster_kwargs)

        gzip_output_folder(
            directory=Path.cwd(),
            setting=SETTINGS.LOBSTER_ZIP_FILES,
            files_list=_FILES_TO_ZIP,
        )

        return LobsterTaskDocument.from_directory(
            Path.cwd(),
            self.additional_fields,
        )
