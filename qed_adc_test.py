# PYTHONPATH must include directory above plugin directory.
#     Define either externally or here, then import plugin.
#sys.path.insert(0, '../..')
#import hilbert
import psi4
import adcc
import hilbert
import numpy as np
import scipy.linalg as sp

#adcc.set_n_threads(4)

# Run SCF in Psi4 
mol = psi4.geometry("""
    H 0 0 0
    F 0 0 0.917 
    symmetry c1
    units au
""")
psi4.set_num_threads(adcc.get_n_threads())
psi4.core.be_quiet()
psi4.set_options({'basis': "sto-3g",
                  'scf_type': 'df'})
psi4.set_module_options('hilbert', {'n_photon_states': 1,
                  'cavity_frequency': '[0.0, 0.0, 0.5]',
                  'cavity_coupling_strength': '[0.0, 0.0, 0.05]'})
scf_e, wfn = psi4.energy('polaritonic-uhf', return_wfn=True)

# Run an adc2 calculation:
refstate = adcc.ReferenceState(wfn)
refstate.coupling = [0.0, 0.0, 0.05]
refstate.frequency = [0.0, 0.0, 0.5]
refstate.qed_hf = True
#refstate.qed_in_matrix = True
#refstate.first_order_coupling = True
state = adcc.adc2(refstate, n_singlets=20)

print(state.describe())
print(state.describe_amplitudes())
print(state.excitation_energy)
#print(state.transition_dipole_moment)
#print(state.s2s_dipole_moment)



coupling = refstate.coupling[2]
freq = refstate.frequency[2]

refstate.second_order_coupling = True

n_adc = len(state.excitation_energy)

# these are just the tdms of one lower order, so qed-adc1 contributions are already included here
qed_adc2_tdm_vec = np.empty(n_adc)

for i, tdm in enumerate(state.transition_dipole_moments_qed):
    qed_adc2_tdm_vec[i] = - np.sqrt(freq/2) * coupling * np.sqrt(2 * freq) * tdm[2]
    
#qed_adc2_tdm_vec *= - np.sqrt(freq/2)

# s2s_dipole parts of the ph_ph blocks
qed_adc1_off_diag_block = - np.sqrt(freq/2) * coupling * np.sqrt(2 * freq) * state.s2s_dipole_moments_qed["qed_adc1_off_diag"]

qed_adc2_diag_block = - np.sqrt(freq/2) * coupling * np.sqrt(2 * freq) * state.s2s_dipole_moments_qed["qed_adc2_diag"]

qed_adc2_diag_block = qed_adc2_diag_block * np.sqrt(freq/2) # missing factor from state.s2s_dipole_moments_qed_adc2_diag

qed_adc2_edge_block_couple = - np.sqrt(freq/2) * coupling * np.sqrt(2 * freq) * state.s2s_dipole_moments_qed["qed_adc2_edge_couple"]

qed_adc2_edge_block_couple = qed_adc2_edge_block_couple * np.sqrt(freq) # missing factor from state.s2s_dipole_moments_qed_adc2_edge

qed_adc2_edge_block_phot_couple = - np.sqrt(freq/2) * coupling * np.sqrt(2 * freq) * state.s2s_dipole_moments_qed["qed_adc2_edge_phot_couple"]

qed_adc2_edge_block_phot_couple = qed_adc2_edge_block_phot_couple * np.sqrt(freq) # missing factor from state.s2s_dipole_moments_qed_adc2_edge


# s2s_dipole parts of the pphh_ph and ph_pphh blocks

qed_adc2_ph_pphh_couple_block = - np.sqrt(freq/2) * coupling * np.sqrt(2 * freq) * state.s2s_dipole_moments_qed["qed_adc2_ph_pphh"]

qed_adc2_pphh_ph_phot_couple_block = - np.sqrt(freq/2) * coupling * np.sqrt(2 * freq) * state.s2s_dipole_moments_qed["qed_adc2_pphh_ph"]


# s2s_dipole parts of off diag pphh_pphh

#qed_adc2_pphh_pphh_off_diag = - np.sqrt(freq/2) * coupling * np.sqrt(2 * freq) * state.s2s_dipole_moments_qed["qed_adc2_pphh_pphh"]


# we still need the H_1 expectation value "as property"

qed_adc2_couple_block = np.sqrt(freq/2) * state.qed_second_order_ph_ph_couplings["couple"]
qed_adc2_phot_couple_block = np.sqrt(freq/2) * state.qed_second_order_ph_ph_couplings["phot_couple"]
print(qed_adc2_couple_block)

# in qed-adc2 omega is only added to the singles block, so we check for the corresponding singles norm

#single_excitation_states = np.zeros(n_adc)

#for i, vec in enumerate(state.excitation_vector):
#    singles_norm = vec.ph.dot(vec.ph)
#    if singles_norm >= 0.8:
#        single_excitation_states[i] = 1

single_excitation_states = np.ones(n_adc)

# now we build the matrix
#qed_adc2_ph_pphh_couple_block = qed_adc2_pphh_ph_phot_couple_block = np.zeros((n_adc, n_adc))
#qed_adc2_diag_block = np.zeros((n_adc, n_adc))
#qed_adc2_edge_block_couple = qed_adc2_edge_block_phot_couple = qed_adc2_diag_block

elec_block = np.diag(state.excitation_energy) + qed_adc2_diag_block

phot_block = np.diag(state.excitation_energy) + qed_adc2_diag_block * 2 + np.diag(single_excitation_states) * freq

phot2_block = np.diag(state.excitation_energy) + qed_adc2_diag_block * 3 + np.diag(single_excitation_states) * 2 * freq

couple_block = qed_adc1_off_diag_block + qed_adc2_ph_pphh_couple_block + qed_adc2_couple_block

phot_couple_block = qed_adc1_off_diag_block + qed_adc2_pphh_ph_phot_couple_block + qed_adc2_phot_couple_block

# edge blocks require no further setup

#couple_inner_block = np.sqrt(2) * couple_block

#phot_couple_inner_block = np.sqrt(2) * phot_couple_block

#qed_adc2_edge_block_couple = qed_adc2_edge_block_phot_couple = np.zeros((n_adc, n_adc))

# there might be an error with the sign of zeroth and first order tdms, since they are different in the implementation of qed-adc2

matrix_1 = np.vstack((elec_block, qed_adc2_tdm_vec.reshape((1, n_adc)), 
                    phot_couple_block, np.zeros((1, n_adc)), qed_adc2_edge_block_phot_couple))
matrix_2 = np.concatenate((qed_adc2_tdm_vec, np.array([freq]),
                         np.zeros(n_adc), np.array([0]), np.zeros(n_adc)))
matrix_3 = np.vstack((couple_block, np.zeros((1, n_adc)), phot_block,
                    np.sqrt(2) * qed_adc2_tdm_vec.reshape((1, n_adc)), np.sqrt(2) * phot_couple_block))
matrix_4 = np.concatenate((np.zeros(n_adc), np.array([0]), np.sqrt(2) * qed_adc2_tdm_vec,
                         2 * np.array([freq]), np.zeros(n_adc)))
matrix_5 = np.vstack((qed_adc2_edge_block_couple, np.zeros((1, n_adc)), 
                    np.sqrt(2) * couple_block, np.zeros((1, n_adc)), phot2_block))

matrix = np.hstack((matrix_1, matrix_2.reshape((len(matrix_2), 1)), matrix_3,
                     matrix_4.reshape((len(matrix_4), 1)), matrix_5))

eigvals, eigvecs = sp.eigh(matrix)

print("energies from initial adc calc with adapted ERIs = ", state.excitation_energy)
print("energies from test = ", eigvals)

#print(qed_adc2_tdm_vec)
#print(qed_adc1_off_diag_block)


#print(elec_block)
#print(qed_adc2_tdm_vec)
#print(qed_adc2_diag_block)


#eigvals, eigvecs = sp.eigh(np.diag(state.excitation_energy) + qed_adc2_diag_block)

#refstate.qed_in_matrix = True
#ref_calc = adcc.adc2(refstate, n_singlets=20)

#print(ref_calc.describe())
#print("eigenvalues from reference = ", ref_calc.excitation_energy)
#print("eigenvalues from test = ", eigvals)
#print("original eigenvalues = ", state.excitation_energy)

"""
tdm_block = np.empty(len(state.excitation_energy))

for i, tdm in enumerate(state.transition_dipole_moment):
    tdm_block[i] = coupling * np.sqrt(2 * freq) * tdm[2]

diffdm_block = - np.sqrt(freq/2) * coupling * np.sqrt(2 * freq) * state.diff_dipole_moment
tdm_block = - np.sqrt(freq/2) * tdm_block

elec_block = np.diag(state.excitation_energy)
phot_block = np.diag(state.excitation_energy + freq)

#print(np.concatenate((tdm_block, np.array([freq]), np.zeros(len(tdm_block)))))

matrix_upper = np.vstack((elec_block, tdm_block.reshape((1, len(tdm_block))), diffdm_block))
matrix_middle = np.concatenate((tdm_block, np.array([freq]), np.zeros(len(tdm_block))))
matrix_lower = np.vstack((diffdm_block, np.zeros((1, len(tdm_block))), phot_block))

matrix = np.hstack((matrix_upper, matrix_middle.reshape((len(matrix_middle), 1)), matrix_lower))

eigvals, eigvecs = sp.eigh(matrix)

print(eigvals)
"""

