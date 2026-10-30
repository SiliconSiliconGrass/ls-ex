"""Analyze SPH fragments from a LS-DYNA d3plot output directory.

Example:
    uv run ls-ex-sph D:\\runs\\case\\d3plot -o outputs\\case-sph
"""

from ls_ex.sph_fragments import main


if __name__ == "__main__":
    raise SystemExit(main())
