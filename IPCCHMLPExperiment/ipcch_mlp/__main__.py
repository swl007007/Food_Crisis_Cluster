from ipcch_mlp.cli import main

if __name__ == "__main__":  # guard: spawned replicate workers re-import the main module
    raise SystemExit(main())
