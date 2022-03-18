#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
import enum
from unicodedata import name
import warnings
import numpy as np

from .misc import cached_member_function, cached_property
from .timings import Timer, timed_member_call
from .visualisation import ExcitationSpectrum
from .OneParticleOperator import OneParticleOperator, product_trace
from .AdcMethod import AdcMethod
from adcc.functions import direct_sum, einsum, zeros_like, ones_like, empty_like
from adcc import block as b
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.adc_pp.transition_dm import transition_dm
from .MoSpaces import MoSpaces
from .ReferenceState import ReferenceState

from scipy import constants
from matplotlib import pyplot as plt
from .Excitation import mark_excitation_property
from .solver.SolverStateBase import EigenSolverStateBase


class ElectronicTransition:
    def __init__(self, data, method=None, property_method=None):
        """Construct an ElectronicTransition class from some data obtained
        from an interative solver or another :class:`ElectronicTransition`
        object.

        Parameters
        ----------
        data
            Any kind of iterative solver state. Typically derived off
            a :class:`solver.EigenSolverStateBase`.
        method : str, optional
            Provide an explicit method parameter if data contains none.
        property_method : str, optional
            Provide an explicit method for property calculations to
            override the automatic selection.
        """
        self.matrix = data.matrix
        self.ground_state = self.matrix.ground_state
        self.reference_state = self.matrix.ground_state.reference_state
        self.operators = self.reference_state.operators

        # List of all the objects which have timers (do not yet collect
        # timers, since new times might be added implicitly at a later point)
        self._property_timer = Timer()
        self._timed_objects = [("", self.reference_state),
                               ("adcmatrix", self.matrix),
                               ("mp", self.ground_state),
                               ("intermediates", self.matrix.intermediates)]
        if hasattr(data, "timer"):
            datakey = getattr(data, "algorithm", data.__class__.__name__)
            self._timed_objects.append((datakey, data))

        # Copy some optional attributes
        for optattr in ["converged", "spin_change", "kind", "n_iter"]:
            if hasattr(data, optattr):
                setattr(self, optattr, getattr(data, optattr))

        self.method = getattr(data, "method", method)
        if self.method is None:
            self.method = self.matrix.method
        if not isinstance(self.method, AdcMethod):
            self.method = AdcMethod(self.method)
        if property_method is None:
            if self.method.level < 3:
                property_method = self.method
            else:
                # Auto-select ADC(2) properties for ADC(3) calc
                property_method = self.method.at_level(2)
        elif not isinstance(property_method, AdcMethod):
            property_method = AdcMethod(property_method)
        self._property_method = property_method

        # Special stuff for special solvers
        if isinstance(data, EigenSolverStateBase):
            self._excitation_vector = data.eigenvectors
            self._excitation_energy_uncorrected = data.eigenvalues
            self.residual_norm = data.residual_norms
        else:
            if hasattr(data, "eigenvalues"):
                self._excitation_energy_uncorrected = data.eigenvalues
            if hasattr(data, "eigenvectors"):
                self._excitation_vector = data.eigenvectors
            # if both excitation_energy and excitation_energy_uncorrected
            # are present, the latter one has priority
            if hasattr(data, "excitation_energy"):
                self._excitation_energy_uncorrected = \
                    data.excitation_energy.copy()
            if hasattr(data, "excitation_energy_uncorrected"):
                self._excitation_energy_uncorrected =\
                    data.excitation_energy_uncorrected.copy()
            if hasattr(data, "excitation_vector"):
                self._excitation_vector = data.excitation_vector

        # Collect all excitation energy corrections
        self._excitation_energy = self._excitation_energy_uncorrected.copy()

    def __len__(self):
        return self.size

    @property
    def size(self):
        return self._excitation_energy.size

    @property
    def timer(self):
        """Return a cumulative timer collecting timings from the calculation"""
        ret = Timer()
        for key, obj in self._timed_objects:
            ret.attach(obj.timer, subtree=key)
        ret.attach(self._property_timer, subtree="properties")
        ret.time_construction = self.reference_state.timer.time_construction
        return ret

    @property
    def property_method(self):
        """The method used to evaluate ADC properties"""
        return self._property_method

    @property
    @mark_excitation_property()
    def excitation_energy(self):
        """Excitation energies including all corrections in atomic units"""
        return self._excitation_energy

    @property
    @mark_excitation_property()
    def excitation_energy_uncorrected(self):
        """Excitation energies without any corrections in atomic units"""
        return self._excitation_energy_uncorrected

    @property
    @mark_excitation_property()
    def excitation_vector(self):
        """List of excitation vectors"""
        return self._excitation_vector

    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def transition_dipole_moment(self):
        """List of transition dipole moments of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition dipole moments are known to be "
                          "faulty in some cases.")
        dipole_integrals = self.operators.electric_dipole
        return np.array([
            [product_trace(comp, tdm) for comp in dipole_integrals]
            for tdm in self.transition_dm
        ])

    @cached_property
    @mark_excitation_property()
    #@timed_member_call(timer="_property_timer")
    def transition_dipole_moments_qed(self):
        """List of transition dipole moments of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition dipole moments are known to be "
                          "faulty in some cases.")
        dipole_integrals = self.operators.electric_dipole
        print("this is the property level", self.property_method.level)
        def tdm(i, prop_level):
            self.ground_state.tdm_contribution = prop_level
            return transition_dm(self.method, self.ground_state, self.excitation_vector[i])
        if hasattr(self.reference_state, "first_order_coupling"):# and self.method.name == "adc2":
            #self.ground_state.tdm_contribution = "adc0"

            single_excitation_states = np.zeros(len(self.excitation_energy))

            for i, vec in enumerate(self.excitation_vector):
                singles_norm = vec.ph.dot(vec.ph)
                if singles_norm >= 0.8:
                    single_excitation_states[i] = 1

            ret = np.zeros((len(self.excitation_energy), 3))
            for i, vec in enumerate(self.excitation_vector):
                if single_excitation_states[i] == 1:
                    ret[i] = np.array([product_trace(comp, tdm(i, "adc0"))# * (vec.ph.dot(vec.ph))**(-1))
                                         for comp in dipole_integrals])
                #else:
                #    ret[i] = np.zeros(3)
            return ret
            #return np.array([
            #    [product_trace(comp, tdm) for comp in dipole_integrals]
            #    for tdm in self.transition_dm
            #])
        else:
            prop_level = "adc" + str(self.property_method.level - 1)
            return np.array([
                [product_trace(comp, tdm(i, prop_level)) for comp in dipole_integrals]
                for i in np.arange(len(self.excitation_energy))
            ])

    @cached_property
    @mark_excitation_property()
    #@timed_member_call(timer="_property_timer")
    def s2s_dipole_moments_qed(self):
        """List of diff_dipole moments of all computed states"""
        dipole_integrals = self.operators.electric_dipole
        print("note, that only the z coordinate of the dipole integrals is calculated")
        n_states = len(self.excitation_energy)
        #print(self.state_diffdm)
        def s2s(i, f, s2s_contribution):
            self.ground_state.s2s_contribution = s2s_contribution
            vec = self.excitation_vector
            return state2state_transition_dm(self.method, self.ground_state, vec[i], vec[f])

        def final_block(name):
            return np.array([[product_trace(dipole_integrals[2], s2s(i, j, name)) for j in np.arange(n_states)]
                     for i in np.arange(n_states)])

        block_dict = {}
        #block = np.zeros((n_states, n_states))
        single_excitation_states = np.zeros(n_states)

        for i, vec in enumerate(self.excitation_vector):
            singles_norm = vec.ph.dot(vec.ph)
            if singles_norm >= 0.8:
                single_excitation_states[i] = 1

        #block = np.outer(single_excitation_states, single_excitation_states)

        #for i in np.arange(n_states):
        #    for j in np.arange(n_states):
        #        #if block[i, j] == 1:
        #        block[i, j] = product_trace(dipole_integrals[2], s2s(i, j, "adc1"))#"qed_adc1"))

        block_dict["qed_adc1_off_diag"] = final_block("adc1")

        if self.method.name == "adc2" and hasattr(self.reference_state, "second_order_coupling"):
            print("second order coupling is calculated as well")
            #for i in np.arange(n_states):
            #    for j in np.arange(n_states): 
            #        block[i, j] = product_trace(dipole_integrals[2], s2s(i, j, "qed_adc2_diag"))
            
            block_dict["qed_adc2_diag"] = final_block("qed_adc2_diag")

            #for i in np.arange(n_states):
            #    for j in np.arange(n_states):
            #        block[i, j] = product_trace(dipole_integrals[2], s2s(i, j, "qed_adc2_edge_couple"))
            
            block_dict["qed_adc2_edge_couple"] = final_block("qed_adc2_edge_couple")

            #for i in np.arange(n_states):
            #    for j in np.arange(n_states):
            #        block[i, j] = product_trace(dipole_integrals[2], s2s(i, j, "qed_adc2_edge_phot_couple"))
            
            block_dict["qed_adc2_edge_phot_couple"] = final_block("qed_adc2_edge_phot_couple")

            #for i in np.arange(n_states):
            #    for j in np.arange(n_states): 
            #        block[i, j] = product_trace(dipole_integrals[2], s2s(i, j, "qed_adc2_ph_pphh"))
            
            block_dict["qed_adc2_ph_pphh"] = final_block("qed_adc2_ph_pphh")

            #for i in np.arange(n_states):
            #    for j in np.arange(n_states): 
            #        block[i, j] = product_trace(dipole_integrals[2], s2s(i, j, "qed_adc2_pphh_ph"))
            
            block_dict["qed_adc2_pphh_ph"] = final_block("qed_adc2_pphh_ph")

            #block_dict["qed_adc2_pphh_pphh"] = final_block("qed_adc2_pphh_pphh")

        return block_dict
        #return np.array([
        #    [product_trace(comp, ddm) for comp in dipole_integrals]
        #    for ddm in self.state_diffdm
        #])

    @cached_property
    @mark_excitation_property()
    def qed_second_order_ph_ph_couplings(self):
        block_dict = {}
        #two_p_op_object = {}
        #omega = self.reference_state.get_qed_omega()
        qed_t1 = self.ground_state.qed_t1(b.ov)
        # check if following objects provide correct symmetry and norm
        # maybe just build p_oo and p_vv and include qed_t1 in final prod_sum
        
        def couple(qed_t1, ul, ur):
            return {
                b.ooov: einsum("kc,ia,ja->kjic", qed_t1, ul, ur) + einsum("ka,ia,jb->jkib", qed_t1, ul, ur),
                b.ovvv: einsum("kc,ia,ib->kacb", qed_t1, ul, ur) + einsum("ic,ia,jb->jabc", qed_t1, ul, ur) 
            }

        def phot_couple(qed_t1, ul, ur):
            return {
                b.ooov: einsum("kc,ia,ja->kijc", qed_t1, ul, ur) + einsum("kb,ia,jb->ikja", qed_t1, ul, ur),
                b.ovvv: einsum("kc,ia,ib->kbca", qed_t1, ul, ur) + einsum("jc,ia,jb->ibac", qed_t1, ul, ur) 
            }

        def prod_sum(hf, two_p_op):
            return - (einsum("ijka,ijka->", hf.ooov, two_p_op[b.ooov]) 
                            + einsum("iabc,iabc->", hf.ovvv, two_p_op[b.ovvv]))
        
        n_states = len(self.excitation_energy)
        single_excitation_states = np.zeros(n_states)

        for i, vec in enumerate(self.excitation_vector):
            singles_norm = vec.ph.dot(vec.ph)
            if singles_norm >= 0.8:
                single_excitation_states[i] = 1

        block_couple = np.outer(single_excitation_states, single_excitation_states)
        block_phot_couple = np.outer(single_excitation_states, single_excitation_states)
        exvec = self.excitation_vector 

        for i in np.arange(n_states):
            for j in np.arange(n_states):
                #if block_couple[i, j] == 1:
                block_couple[i, j] = prod_sum(self.reference_state, couple(qed_t1, exvec[i].ph, exvec[j].ph))
                block_phot_couple[i, j] = prod_sum(self.reference_state, phot_couple(qed_t1, exvec[i].ph, exvec[j].ph))

        block_dict["couple"] = block_couple
        block_dict["phot_couple"] = block_phot_couple
        """
        def s2s(i, f):#, s2s_contribution):
            self.ground_state.s2s_contribution = "adc0"
            vec = self.excitation_vector
            return state2state_transition_dm(self.method, self.ground_state, vec[i], vec[f])    

        def prod_sum_couple(hf, i, f):
            return 0.25 * (
                - einsum("kc,ji,kjic->", qed_t1, s2s(i, f).oo, hf.ooov)
                - einsum("ka,ia,jb,jkib->", ) # we probably need an other factor, and we need to check the norm and symmetry
            )
        """
        return block_dict

    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def transition_dipole_moment_velocity(self):
        """List of transition dipole moments in the
        velocity gauge of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition velocity dipole moments "
                          "are known to be faulty in some cases.")
        dipole_integrals = self.operators.nabla
        return np.array([
            [product_trace(comp, tdm) for comp in dipole_integrals]
            for tdm in self.transition_dm
        ])

    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def transition_magnetic_dipole_moment(self):
        """List of transition magnetic dipole moments of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition magnetic dipole moments "
                          "are known to be faulty in some cases.")
        mag_dipole_integrals = self.operators.magnetic_dipole
        return np.array([
            [product_trace(comp, tdm) for comp in mag_dipole_integrals]
            for tdm in self.transition_dm
        ])

    @cached_property
    @mark_excitation_property()
    def oscillator_strength(self):
        """List of oscillator strengths of all computed states"""
        #print("energies =", self.excitation_energy.tolist())
        #print("eigenvectors =", self.excitation_vector[0].ph.to_ndarray())# for vec in self.excitation_vector])
        #print("tdms =", self.transition_dipole_moment.tolist())
        #print("transition_dm", [trans_dm.to_ndarray().shape for trans_dm in self.transition_dm])
        #print("dipole integrals", [el_dip.oo.to_ndarray().shape for el_dip in self.operators.electric_dipole])
        #print("state_diffdm", [val.blocks for val in self.state_diffdm])#[diffdm.evaluate() for diffdm in self.state_diffdm])
        #print("state_diffdm has oo and vv ?", self.state_diffdm.oo, self.state_diffdm.vv)

        #def s2s(state_i, state_f, space):
        #    if space == "ov":
        #        return state2state_transition_dm(self.method, self.ground_state, self.excitation_vector[state_i], self.excitation_vector[state_f]).ov
        #    elif space == "oo":
        #        return state2state_transition_dm(self.method, self.ground_state, self.excitation_vector[state_i], self.excitation_vector[state_f]).oo
        #    elif space == "vv":
        #        return state2state_transition_dm(self.method, self.ground_state, self.excitation_vector[state_i], self.excitation_vector[state_f]).vv
        #    else:
        #        raise AttributeError("OneParticle operator object has no attribute {f}", space)
        """
        def s2s(i, f):
            return state2state_transition_dm(self.method, self.ground_state, self.excitation_vector[i], self.excitation_vector[f])

        #print("s2s", s2s(0, 1, "oo"))
        #print("s2s", s2s(1, 0, "oo"))

        total_dip = OneParticleOperator(self.reference_state.mospaces, is_symmetric=True)
        from . import block as b
        total_dip.oo = ReferenceState.get_qed_total_dip(self.reference_state, b.oo)
        total_dip.vv = ReferenceState.get_qed_total_dip(self.reference_state, b.vv)
        total_dip.ov = ReferenceState.get_qed_total_dip(self.reference_state, b.ov)

        off_diag_block = np.empty((len(self.excitation_energy), len(self.excitation_energy)))
        #off_diag_block = np.array([[[product_trace(comp, s2s(i, f)) for comp in self.operators.electric_dipole]
        #                            for i in np.arange(len(self.excitation_energy))]
        #                            for f in np.arange(len(self.excitation_energy))])
        ovov = ReferenceState.qed_D_object(self.reference_state, b.ovov)
        np.save("/home/marco/D_ovov", ovov.to_ndarray())

        #dip0 = 0
        #qed_coupls, qed_freqs = ReferenceState.get_qed_params(self.reference_state)
        print("Warning: The groundstate dipole moment is calculated at the mp1 level")
        #for coupling, freq, dip in zip(qed_coupls, qed_freqs, self.ground_state.dipole_moment(1)):
        #    dip0 += coupling * np.sqrt(2 * freq) * dip
        

        for i in np.arange(len(self.excitation_energy)):
            for j in np.arange(len(self.excitation_energy)):
                off_diag_block[i, j] = product_trace(total_dip, s2s(i, j))
                #if i == j:
                #    off_diag_block[i, j] = dip0 - product_trace(total_dip, s2s(i, j))
                #else:
                #    off_diag_block[i, j] = product_trace(total_dip, s2s(i, j))
                
        #tdm_arr = np.empty(len(self.transition_dipole_moment))

        #print([product_trace(total_dip, trans_dm) for trans_dm in self.transition_dm])

        #print(off_diag_block)
        #np.save("/home/marco/off_diag_isr_basis", off_diag_block)
        print("off_diag_block = ", list(off_diag_block))
        
        #print((self.ground_state.dipole_moment(1) * 0.05 - off_diag_block[0, 0]) * 20)
        """



        """
        diff_dip = OneParticleOperator(self.reference_state.mospaces, is_symmetric=True)
        total_dip = OneParticleOperator(self.reference_state.mospaces, is_symmetric=True)
        from . import block as b
        total_dip.oo = ReferenceState.get_qed_total_dip(self.reference_state, b.oo)
        total_dip.vv = ReferenceState.get_qed_total_dip(self.reference_state, b.vv)

        d_oo = zeros_like(total_dip.oo)
        d_vv = zeros_like(total_dip.vv)
        d_oo.set_mask("ii", 1.0)
        d_vv.set_mask("aa", 1.0)

        ds_init = OneParticleOperator(self.reference_state.mospaces, is_symmetric=True) #Since there is no TwoParticleOperator we do this
        ds = {
            #b.oooo: einsum('ik,jl->ijkl', ds_init.oo, ds_init.oo),
            #b.ooov: einsum('ik,ja->ijka', ds_init.oo, ds_init.ov),
            #b.oovv: einsum('ia,jb->ijab', ds_init.ov, ds_init.ov),
            #b.ovvv: einsum('ib,ac->iabc', ds_init.ov, ds_init.vv),
            b.ovov: einsum('ij,ab->iajb', ds_init.oo, ds_init.vv),
            #b.vvvv: einsum('ac,bd->abcd', ds_init.vv, ds_init.vv),
        }

        ds[b.ovov] = einsum("ij,ab->iajb", d_oo, total_dip.vv) - einsum("ij,ab->iajb", total_dip.oo, d_vv)

        #print(ds[b.ovov])

        np.save("/home/marco/diff_dens_dip_non_transformed", ds[b.ovov].to_ndarray())
        np.save("/home/marco/diff_dens_dip_oo_non_transformed", einsum("ij,ab->iajb", total_dip.oo, d_vv).to_ndarray())
        np.save("/home/marco/diff_dens_dip_vv_non_transformed", einsum("ij,ab->iajb", d_oo, total_dip.vv).to_ndarray())
        np.save("/home/marco/eigvecs_adc1", [vec.ph.to_ndarray() for vec in self.excitation_vector])

        diff_dips = {}

        #for i in np.arange(len(self.excitation_vector)):
        #    for j in np.arange(i + 1):
        """
                


        """
        off_diag_block = np.empty([len(self.excitation_vector), 3, *diff_dip.ov.to_ndarray().shape])
        for i, vec in enumerate(self.excitation_vector):
            for j, el_dip in enumerate(self.operators.electric_dipole):
                diff_dip.oo = self.state_diffdm[i].oo * el_dip.oo
                diff_dip.vv = self.state_diffdm[i].vv * el_dip.vv
                diff_dip.ov = einsum("ab,ib->ia", diff_dip.vv, vec.ph) - einsum("ij,ja->ia", diff_dip.oo, vec.ph)
                off_diag_block[i][j] = diff_dip.ov.to_ndarray()
        print(off_diag_block[0])
        """
        #np.save("/home/marco/off_diag_block_adc1", off_diag_block)
        #print(diff_dip.ov.to_ndarray())
        #np_vec = self.excitation_vector[0].ph.to_ndarray()
        #ampl_vec = ones_like(self.excitation_vector[0].ph)
        #print(ampl_vec[0])
        #print(self.excitation_vector[0].ph)
        #print(self.excitation_vector[0].values())
        #diff_dip_ij = zeros_like(self.transition_dm[0].oo)
        #diff_dip_ij.oo = self.transition_dm[0].oo * self.operators.electric_dipole[0].oo
        #diff_dip_ab = zeros_like(self.transition_dm[0].vv)
        #diff_dip_ab.vv = self.transition_dm[0].vv * self.operators.electric_dipole[0].vv
        #diff_dip_ia = zeros_like(self.excitation_vector)
        #diff_dip_ia.ov = einsum("ab,ib->ia", diff_dip_ab.vv, self.excitation_vector[0].ov) - einsum("ij,ja->ia", diff_dip_ij.oo, self.excitation_vector[0].ov)
        #print("diff_dip_ia", diff_dip_ia.ov.to_ndarray())
        #print(product_trace(self.state_diffdm[0], self.operators.electric_dipole[0]))
        #check = self.state_diffdm[0].oo.dot(self.operators.electric_dipole[0].oo)
        #check += self.state_diffdm[0].vv.dot(self.operators.electric_dipole[0].vv)
        #print(check)
        #trans_with_dip = self.transition_dm[0].ov.dot(self.operators.electric_dipole[0].ov) #einsum("ia,ia->ia", self.transition_dm[0].ov, self.operators.electric_dipole[0].ov)
        #trans_with_dip += self.transition_dm[0].vo.dot(self.operators.electric_dipole[0].ov.transpose())
        #print(trans_with_dip)
        #print(self.transition_dm[0].blocks, self.operators.electric_dipole[0].blocks)
        #print(self.state_diffdm[0].is_symmetric)
        #print("hopefully equal to tmd", np.sum(trans_with_dip.to_ndarray()) )#einsum("ia,ia->", trans_with_dip, ones_like(trans_with_dip)))
        return 2. / 3. * np.array([
            np.linalg.norm(tdm)**2 * np.abs(ev)
            for tdm, ev in zip(self.transition_dipole_moment,
                               self.excitation_energy)
        ])

    @cached_property
    @mark_excitation_property()
    def oscillator_strength_velocity(self):
        """List of oscillator strengths in
        velocity gauge of all computed states"""
        return 2. / 3. * np.array([
            np.linalg.norm(tdm)**2 / np.abs(ev)
            for tdm, ev in zip(self.transition_dipole_moment_velocity,
                               self.excitation_energy)
        ])

    @cached_property
    @mark_excitation_property()
    def rotatory_strength(self):
        """List of rotatory strengths of all computed states"""
        return np.array([
            np.dot(tdm, magmom) / ee
            for tdm, magmom, ee in zip(self.transition_dipole_moment_velocity,
                                       self.transition_magnetic_dipole_moment,
                                       self.excitation_energy)
        ])

    @property
    @mark_excitation_property()
    def cross_section(self):
        """List of one-photon absorption cross sections of all computed states"""
        # TODO Source?
        fine_structure = constants.fine_structure
        fine_structure_au = 1 / fine_structure
        prefac = 2.0 * np.pi ** 2 / fine_structure_au
        return prefac * self.oscillator_strength

    def plot_spectrum(self, broadening="lorentzian", xaxis="eV",
                      yaxis="cross_section", width=0.01, **kwargs):
        """One-shot plotting function for the spectrum generated by all states
        known to this class.

        Makes use of the :class:`adcc.visualisation.ExcitationSpectrum` class
        in order to generate and format the spectrum to be plotted, using
        many sensible defaults.

        Parameters
        ----------
        broadening : str or None or callable, optional
            The broadening type to used for the computed excitations.
            A value of None disables broadening any other value is passed
            straight to
            :func:`adcc.visualisation.ExcitationSpectrum.broaden_lines`.
        xaxis : str
            Energy unit to be used on the x-Axis. Options:
            ["eV", "au", "nm", "cm-1"]
        yaxis : str
            Quantity to plot on the y-Axis. Options are "cross_section",
            "osc_strength", "dipole" (plots norm of transition dipole),
            "rotational_strength" (ECD spectrum with rotational strength)
        width : float, optional
            Gaussian broadening standard deviation or Lorentzian broadening
            gamma parameter. The value should be given in atomic units
            and will be converted to the unit of the energy axis.
        """
        if xaxis == "eV":
            eV = constants.value("Hartree energy in eV")
            energies = self.excitation_energy * eV
            width = width * eV
            xlabel = "Energy (eV)"
        elif xaxis in ["au", "Hartree", "a.u."]:
            energies = self.excitation_energy
            xlabel = "Energy (au)"
        elif xaxis == "nm":
            hc = constants.h * constants.c
            Eh = constants.value("Hartree energy")
            energies = hc / (self.excitation_energy * Eh) * 1e9
            xlabel = "Wavelength (nm)"
            if broadening is not None and not callable(broadening):
                raise ValueError("xaxis=nm and broadening enabled is "
                                 "not supported.")
        elif xaxis in ["cm-1", "cm^-1", "cm^{-1}"]:
            towvn = constants.value("hartree-inverse meter relationship") / 100
            energies = self.excitation_energy * towvn
            width = width * towvn
            xlabel = "Wavenumbers (cm^{-1})"
        else:
            raise ValueError("Unknown xaxis specifier: {}".format(xaxis))

        if yaxis in ["osc", "osc_strength", "oscillator_strength", "f"]:
            absorption = self.oscillator_strength
            ylabel = "Oscillator strengths (au)"
        elif yaxis in ["dipole", "dipole_norm", "μ"]:
            absorption = np.linalg.norm(self.transition_dipole_moment, axis=1)
            ylabel = "Modulus of transition dipole (au)"
        elif yaxis in ["cross_section", "σ"]:
            absorption = self.cross_section
            ylabel = "Cross section (au)"
        elif yaxis in ["rot", "rotational_strength", "rotatory_strength"]:
            absorption = self.rotatory_strength
            ylabel = "Rotatory strength (au)"
        else:
            raise ValueError("Unknown yaxis specifier: {}".format(yaxis))

        sp = ExcitationSpectrum(energies, absorption)
        sp.xlabel = xlabel
        sp.ylabel = ylabel
        if not broadening:
            plots = sp.plot(style="discrete", **kwargs)
        else:
            kwdisc = kwargs.copy()
            kwdisc.pop("label", "")
            plots = sp.plot(style="discrete", **kwdisc)

            kwargs.pop("color", "")
            sp_broad = sp.broaden_lines(width, shape=broadening)
            plots.extend(sp_broad.plot(color=plots[0].get_color(),
                                       style="continuous", **kwargs))

        if xaxis in ["nm"]:
            # Invert x axis
            plt.xlim(plt.xlim()[::-1])
        return plots
