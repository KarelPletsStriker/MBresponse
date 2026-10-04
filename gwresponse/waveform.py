"""
To calculate de response of the GW we'll use the PhenomD model as implemented in the JAX-based
package ripple (https://github.com/GW-JAX-Team/ripple)
"""


import jax

from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np

from ripplegw.constants import MTSUN
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD import Phase, IMRPhenDAmplitude_NoCut, get_Amp0, get_IIb_raw_phase
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_utils import get_coeffs, get_transition_frequencies
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_QNMdata import fM_CUT

from . import globals as glob


##############################################################################
# IMRPhenomD WAVEFORM
##############################################################################


class IMRPhenomD_JAX():
    """
    IMRPhenomD waveform model, wrapping the implementation of ripple.

    Relevant references:
        [1] `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_

        [2] `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>`_

        [3] `arXiv:2302.05329 <https://arxiv.org/abs/2302.05329>`_ (ripple)

    :param float, optional fRef: Reference frequency of the waveform, in :math:`\\rm Hz`. If not provided, the minimum of the frequency grid will be used.
    :param bool, optional apply_fcut: If ``True``, the amplitude is set to zero and the phase is kept constant above the cut frequency :math:`Mf = 0.2`.

    """
    def __init__(self, objType = 'BBH', fRef=None, apply_fcut=True, **kwargs):
        """
        Constructor method
        """
        # The kind of system the wf model is made for, can be 'BBH', 'BNS' or 'NSBH'
        self.objType = objType
        # Dimensionless frequency (Mf) at which we define the end of the waveform
        self.fcutPar = fM_CUT
        self.apply_fcut = apply_fcut

        # Dictionary containing the order in which the parameters will appear in the Fisher matrix
        self.ParNums = {'Mc':0, 'eta':1, 'dL':2, 'theta':3, 'phi':4, 'iota':5, 'psi':6, 'tcoal':7, 'Phicoal':8, 'chiS':9,  'chiA':10}
        self.ParNums = dict(sorted(self.ParNums.items(), key=lambda item: item[1]))

        # Dimensionless frequencies (Mf) of the merger-ringdown phase join and of the amplitude peak, filled by Phi and Ampl
        self.PHI_fjoin_MRD = None
        self.fpeak = None

        self.fRef = fRef

    @staticmethod
    def _ripple_inputs(Mc, eta, chi1, chi2):
        """
        Build the inputs of the ripple IMRPhenomD functions for a single event.

        :return: intrinsic parameters ``[m1, m2, chi1, chi2]``, the phenomenological coefficients, the transition frequencies (in :math:`\\rm Hz`) and the total mass in seconds.
        """
        m1, m2 = Mc_eta_to_ms(np.array([Mc, eta]))
        theta = np.array([m1, m2, chi1, chi2])
        coeffs = get_coeffs(theta)
        transition_freqs = get_transition_frequencies(theta, coeffs[5], coeffs[6])
        M_s = (m1 + m2) * MTSUN
        return theta, coeffs, transition_freqs, M_s

    def Phi(self, f, **kwargs):
        """
        Compute the phase of the GW as a function of frequency, given the events parameters.

        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: GW phase for the chosen events evaluated on the frequency grid, and its derivative with respect to the dimensionless frequency :math:`Mf`.
        :rtype: tuple(array, array)

        """
        f = np.asarray(f)

        def single_event(Mc, eta, chi1, chi2):
            theta, coeffs, transition_freqs, M_s = self._ripple_inputs(Mc, eta, chi1, chi2)
            _, _, _, f4, f_RD, f_damp = transition_freqs

            # Time shift so that peak amplitude is approximately at t=0
            t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, theta, coeffs, f_RD, f_damp)

            # Set fRef as the minimum frequency
            fRef = np.amin(f) if self.fRef is None else self.fRef
            phiRef = Phase(fRef, theta, coeffs, transition_freqs)

            def phase(fr):
                # Above the cut frequency the phase is kept constant
                if self.apply_fcut:
                    fr = np.minimum(fr, self.fcutPar / M_s)
                return Phase(fr, theta, coeffs, transition_freqs) - t0 * M_s * (fr - fRef) - phiRef

            # The phase is local in frequency, so a JVP with unit tangent gives d(phase)/df on the whole grid
            phi, dphi_df = jax.jvp(phase, (f,), (np.ones_like(f),))
            # Convert to d(phase)/d(Mf), using the same mass-to-seconds factor as the response
            M_s_glob = M_s / MTSUN * glob.GMsun_over_c3

            return phi, dphi_df / M_s_glob, f_RD * M_s_glob

        phi, dphi, fring = jax.vmap(single_event)(kwargs['Mc'], kwargs['eta'], kwargs['chi1z'], kwargs['chi2z'])
        self.PHI_fjoin_MRD = fring

        return phi[0], dphi[0]

    def Ampl(self, f, **kwargs):
        """
        Compute the amplitude of the GW as a function of frequency, given the events parameters.

        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the amplitude of, as in :py:data:`events`.
        :return: GW amplitude for the chosen events evaluated on the frequency grid.
        :rtype: array

        """
        f = np.asarray(f)

        def single_event(Mc, eta, chi1, chi2, dL):
            theta, coeffs, transition_freqs, M_s = self._ripple_inputs(Mc, eta, chi1, chi2)
            fpeak = transition_freqs[3]

            # ripple's own cut assumes a uniform frequency grid, so use the uncut amplitude and apply the cut here
            amplitudeIMR = IMRPhenDAmplitude_NoCut(f, theta, coeffs, transition_freqs)
            if self.apply_fcut:
                amplitudeIMR = np.where(f * M_s < self.fcutPar, amplitudeIMR, 0.)

            # Defined as in LALSimulation - LALSimIMRPhenomD.c line 332. Final units are correctly Hz^-1
            Overallamp = 2. * np.sqrt(5./(64.*np.pi)) * M_s * M_s * glob.c / (dL * glob.uGpc)

            return Overallamp * get_Amp0(f * M_s, eta) * amplitudeIMR, fpeak * M_s

        ampl, fpeak = jax.vmap(single_event)(kwargs['Mc'], kwargs['eta'], kwargs['chi1z'], kwargs['chi2z'], kwargs['dL'])
        self.fpeak = fpeak

        return ampl[0]

    def tau_star(self, f, **kwargs):
        """
        Compute the time to coalescence (in seconds) as a function of frequency (in :math:`\\rm Hz`), given the events parameters.

        We use the expression in `arXiv:0907.0700 <https://arxiv.org/abs/0907.0700>`_ eq. (3.8b).

        :param array f: Frequency grid on which the time to coalescence will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
        :return: time to coalescence for the chosen events evaluated on the frequency grid, in seconds.
        :rtype: array

        """
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta

        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))

        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (-10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)

        return OverallFac*(t05 + t6 + t7)

    def fcut(self, **kwargs):
        """
        Compute the cut frequency of the waveform as a function of the events parameters, in :math:`\\rm Hz`.

        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the cut frequency of, as in :py:data:`events`.
        :return: Cut frequency of the waveform for the chosen events, in :math:`\\rm Hz`.
        :rtype: array

        """
        return self.fcutPar/(kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.)))
