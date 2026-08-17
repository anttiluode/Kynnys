from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

from experiments import gate4b_noise_corrected as gate

# Gate 4b's experiment body is unchanged; inject the missing standard-library
# symbol that the result formatter references.
gate.median = median


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("gate4b_result.json"))
    args = p.parse_args()
    result = gate.run(args.output)
    if not result["gate"]["pass"]:
        raise SystemExit("Gate 4b did not pass its preregistered criteria")


if __name__ == "__main__":
    main()
