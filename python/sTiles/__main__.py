"""Command line entry point: python -m sTiles <command>

    python -m sTiles info       which solver is loaded, and from where
    python -m sTiles update     re-download the current released solver
    python -m sTiles clean      delete every cached solver
"""
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "info"
    import sTiles

    if cmd == "info":
        print("package:", sTiles.__version__)
        print("solver: ", sTiles.version())
        print("loaded: ", sTiles.library_path())
    elif cmd == "update":
        lib = sTiles.update()
        print("cached:", lib)
        print("restart Python for it to take effect (the solver loads once per process).")
    elif cmd == "clean":
        n = sTiles.clear_cache()
        print(f"removed {n} cached solver(s); the next import downloads the current one.")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
