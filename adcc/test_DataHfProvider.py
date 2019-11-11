#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import os
import tempfile
import unittest
import numpy as np

from adcc.test_ReferenceState_refdata import compare_refstate_with_reference
from adcc.testdata.cache import cache

import h5py


class TestDataHfProvdier(unittest.TestCase):
    def test_dict(self):
        data = cache.hfdata["cn_sto3g"]
        spec = "gen"
        refdata = cache.hfimport["cn_sto3g"][spec]

        if "threshold" in data and "conv_tol" not in data:
            # Fall back to deprecated key threshold
            data["conv_tol"] = data["threshold"]

        bdict = dict()
        for key in ["restricted", "conv_tol", "occupation_f", "orbcoeff_fb",
                    "orben_f", "fock_ff", "eri_ffff", "energy_scf",
                    "spin_multiplicity"]:
            bdict[key] = data[key]

        dmmp = data["multipoles"]
        bdict["multipoles"] = {
            "elec_1": np.asarray(dmmp["elec_1"]),
            "nuclear_0": dmmp["nuclear_0"],
            "nuclear_1": dmmp["nuclear_1"],
        }

        # Import hfdata from dict
        compare_refstate_with_reference(data, refdata, spec, scfres=bdict,
                                        compare_eri="abs")

    def test_hdf5(self):
        data = cache.hfdata["cn_sto3g"]
        spec = "gen"
        refdata = cache.hfimport["cn_sto3g"][spec]

        if "threshold" in data and "conv_tol" not in data:
            # Fall back to deprecated key threshold
            data["conv_tol"] = data["threshold"]

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "data.hdf5")
            with h5py.File(fn, "w") as h5f:
                h5f.create_dataset("restricted", data=data["restricted"])
                h5f.create_dataset("conv_tol", data=data["conv_tol"])
                h5f.create_dataset("occupation_f", data=data["occupation_f"])
                h5f.create_dataset("orbcoeff_fb", data=data["orbcoeff_fb"])
                h5f.create_dataset("orben_f", data=data["orben_f"])
                h5f.create_dataset("fock_ff", data=data["fock_ff"])
                h5f.create_dataset("eri_ffff", data=data["eri_ffff"])

                # Optional keys
                h5f.create_dataset("energy_scf", data=data["energy_scf"])
                h5f.create_dataset("spin_multiplicity",
                                   data=data["spin_multiplicity"])

                dmmp = data["multipoles"]
                mmp = h5f.create_group("multipoles")
                mmp.create_dataset("elec_1", data=np.asarray(dmmp["elec_1"]))
                mmp.create_dataset("nuclear_0", data=dmmp["nuclear_0"])
                mmp.create_dataset("nuclear_1", data=dmmp["nuclear_1"])

            # Import hfdata from hdf5 file
            compare_refstate_with_reference(data, refdata, spec, scfres=fn,
                                            compare_eri="abs")
