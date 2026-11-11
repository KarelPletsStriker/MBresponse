import logging

import jax
import pytest

jax.config.update("jax_enable_x64", True)
logging.getLogger("jimgw").setLevel(logging.WARNING)

from jimgw.core.single_event.waveform import RippleIMRPhenomD  # noqa: E402

GPS = 1.4e9

#: A massive black hole binary in LISA, on a short, cheap segment
LISA_PARAMS = {
    "M_c": 4.0e5, "eta": 0.24, "s1_z": 0.3, "s2_z": -0.1, "d_L": 3.0e4, "iota": 0.5,
    "phase_c": 0.3, "ra": 1.0, "dec": 0.3, "psi": 0.7, "t_c": 50.0,
}
LISA_DATA = {
    "duration": 2.0**15, "sampling_frequency": 0.1, "trigger_time": GPS,
    "f_min": 3e-3, "f_max": 0.05, "post_trigger_duration": 6e3,
}


@pytest.fixture(scope="session")
def waveform():
    return RippleIMRPhenomD(f_ref=3e-3)


def make_lisa(waveform, params=LISA_PARAMS, channels="AE", noise_key=None, **kwargs):
    from gwresponse import get_LISA

    ifos = get_LISA(channels, trigger_time=GPS, **kwargs)
    for ifo in ifos:
        ifo.load_and_set_psd()
        if noise_key is None:
            ifo.set_zero_noise(**LISA_DATA)
        else:
            ifo.simulate_and_set_noise(**LISA_DATA, rng_key=noise_key)
        ifo.inject_signal(waveform, params)
    return ifos
