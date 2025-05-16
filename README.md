# Multilevel Monte Carlo (MLMC)

Efficient Monte Carlo estimation using multilevel sampling, based on the method introduced by Mike Giles (2008).

---

## Project Structure

| Folder      | Contents                                                                 |
|-------------|--------------------------------------------------------------------------|
| `mlmc/`     | Core MLMC implementation (algorithm, Black-Scholes models, payoff functions) |
| `tests/`    | Simple tests to validate MLMC behavior                                   |
| `examples/` | Ready-to-run examples                                                   |
| `summary/`  | Theoretical background and technical notes from academic papers         |

---

## Quick start

Clone the repository and set up a local environment:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate     # (or .venv\\Scripts\\activate on Windows)

# Install dependencies
pip install -r requirements.txt
