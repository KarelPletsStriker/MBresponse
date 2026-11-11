# MBresponse

Two independent Python packages for LISA data analysis, kept in one repository:

| Directory | Package | What it does |
|---|---|---|
| [`galactic-foreground/`](galactic-foreground/) | `galactic_foreground` | Galactic double-white-dwarf population: mass distribution, pairing, mass-transfer stability, frequency evolution, and a synthetic catalogue generator for the LISA foreground. A Python port of Lucy McNeill's notebook. |
| [`lisa-response/`](lisa-response/) | `gwresponse` | LISA frequency-domain TDI response (A, E, T) as a [Jim](https://github.com/GW-JAX-Team/Jim) detector, analytic instrument noise, and joint LISA + ground-based likelihoods for parameter estimation with IMRPhenomD. |

The packages do not import each other; each has its own `pyproject.toml`, tests and README.

```bash
git clone https://github.com/KarelPletsStriker/MBresponse.git
cd MBresponse
pip install -e "lisa-response[test,examples]"   # response and analysis
pip install -e "galactic-foreground[test]"      # population generation
pytest lisa-response/tests
pytest galactic-foreground/tests
```
