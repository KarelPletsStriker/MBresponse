from setuptools import setup, find_packages

setup(
    name='gwresponse',
    version='0.1.0',
    packages=find_packages(),
    description='LISA waveform and response modeling toolkit',
    author='Mireia Egido',
    python_requires='>=3.11',
    install_requires=[
        'jax>=0.8.2',
        'jaxlib>=0.8.2',
        'ripplegw',
        'jimgw',
        'numpy',
        'matplotlib',
        'ipykernel'],
    include_package_data=True,
    package_data={
        'gwresponse': ['WFfiles/*.txt', 'WFfiles/*.h5'],
    },
)