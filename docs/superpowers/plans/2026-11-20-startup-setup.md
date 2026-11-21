# Startup setup implementation plan

**Goal** Gate MangaLens startup on all required libraries and models and let users install the missing items from one window.

**Design** The window uses Tk before application imports. It scans at each launch, shows the current check and 0–100%, lists missing libraries and models separately, and offers selection, select all, and an install action for each list. It rescans after installation and opens the app only when everything required passes. Model downloads report measured bytes, retain partial files for supported resumes, and transient connection errors retry automatically with a manual retry action. Library installation reports its current stage. Setup never starts a download merely because it finds a missing item.

## Work assignments

- [x] Inventory live and main Python dependencies and implement isolated library detection and selected installation in `core/setup_libraries.py` with focused tests.
- [x] Inventory model assets and implement offline detection, downloads, progress, and resumption in `core/setup_models.py` with focused tests.
- [x] Implement the setup window, scan, selection, background work, retries, and handoff in `frontend/setup_window.py` with focused tests.
- [x] Integrate `main.py` using a shared contract in `core/setup_types.py`, reconcile runtime configuration, verify startup and recovery, and update installation instructions.

## Contract and checks

`InstallationItem` has `key`, `label`, `category`, `is_ready()` and `install(progress)` fields. Progress accepts completed bytes and optional total bytes. Retryable network failures raise `DownloadInterrupted`. Setup is read-only during scans. Multiple versions of the same library are checked against their own target. An offline launch with installed files needs no network. A failed or partial install may never unlock startup.
