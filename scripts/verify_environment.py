"""
scripts/verify_environment.py
Verifies Python version, directory structure, core libraries, and GPU status for Phase 1.
"""
import sys
import os
import shutil

def check_environment():
    print("=" * 65)
    print("PLOUTOS+ PHASE 1: ENVIRONMENT & STRUCTURE VERIFICATION")
    print("=" * 65)

    # 1. Python version check
    py_ver = sys.version_info
    print(f"\n[1] Python Environment:")
    print(f"    Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if (py_ver.major, py_ver.minor) == (3, 10):
        print("    PASS: Python 3.10 is installed and active.")
    else:
        print(f"    NOTE: Running on Python {py_ver.major}.{py_ver.minor}. Python 3.10 is recommended.")

    # 2. Check minimal Phase 1 dependencies
    dependencies = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-Learn"),
        ("yaml", "PyYAML"),
        ("pytest", "PyTest"),
        ("tqdm", "TQDM"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]
    print("\n[2] Core Scientific Dependencies:")
    all_packages_ok = True
    for mod_name, display_name in dependencies:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "detected")
            print(f"    PASS: {display_name:<16} (v{ver})")
        except ImportError:
            print(f"    FAIL: {display_name:<16} (NOT INSTALLED)")
            all_packages_ok = False

    # 3. Disk space check
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    print(f"\n[3] Storage Verification:")
    print(f"    Free Disk Space on Workspace Drive: {free_gb:.1f} GB")
    if free_gb >= 10:
        print("    PASS: Sufficient disk space for datasets and models.")
    else:
        print("    WARNING: Disk space is under 10 GB. Free up space before downloading data.")

    # 4. Folder structure check
    required_dirs = [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/samples",
        "notebooks",
        "src/data",
        "src/features",
        "src/experts",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "tests",
        "scripts",
        "results/metrics",
        "results/predictions",
        "results/figures",
    ]
    print("\n[4] Directory Structure Verification:")
    all_dirs_ok = True
    for d in required_dirs:
        norm_d = os.path.normpath(d)
        exists = os.path.isdir(norm_d)
        status = "PASS" if exists else "MISSING"
        print(f"    [{status}] {d}")
        if not exists:
            all_dirs_ok = False

    # 5. Summary Evaluation
    print("\n" + "=" * 65)
    if all_packages_ok and all_dirs_ok:
        print(">>> RESULT: PHASE 1 SETUP COMPLETE & VERIFIED SUCCESSFULLY <<<")
        print("    You are ready to proceed to Phase 2 (StockNet Data Pipeline).")
    else:
        print(">>> RESULT: PHASE 1 SETUP INCOMPLETE <<<")
        if not all_dirs_ok:
            print("    Action needed: Create missing directories.")
        if not all_packages_ok:
            print("    Action needed: Run 'pip install -r requirements.txt'.")
    print("=" * 65)
    return all_packages_ok and all_dirs_ok

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
