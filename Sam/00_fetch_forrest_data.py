import subprocess

#S Torrisi + mostly Claude, for NeuroHackademy '26, Study Forrest project.
#First script for fetching the data from github


# --- Step 1: build file lists ---
subjects = ["01","02","03","04","05","06","09","10","14","15","16","17","18","19","20"]
runs = [1, 2, 3, 4]

aligned_files = []
phase2_files = []
templatetransform_files = []

for sub in subjects:
    templatetransform_files.append(f"sub-{sub}/bold3Tp2/brain_mask.nii.gz")
    for run in runs:
        aligned_files.append(f"sub-{sub}/in_bold3Tp2/sub-{sub}_task-objectcategories_run-{run}_bold.nii.gz")
        aligned_files.append(f"sub-{sub}/in_bold3Tp2/sub-{sub}_task-objectcategories_run-{run}_bold_mcparams.txt")
        phase2_files.append(f"sub-{sub}/ses-localizer/func/sub-{sub}_ses-localizer_task-objectcategories_run-{run}_events.tsv")

repo_base = "/home/jovyan"

# --- Step 2: preview ---
def preview_missing(repo_name, files):
    repo_dir = f"{repo_base}/{repo_name}"
    result = subprocess.run(
        ["git", "annex", "find", "--not", "--in", "here"] + files,
        capture_output=True, text=True, cwd=repo_dir
    )
    if result.returncode != 0:
        print(f"{repo_dir}: ERROR — {result.stderr.strip()}")
        return []
    missing = result.stdout.strip().splitlines()
    print(f"{repo_name}: {len(missing)} of {len(files)} files missing")
    return missing

missing_phase2 = preview_missing("studyforrest-data-phase2", phase2_files)
missing_tt = preview_missing("studyforrest-data-templatetransforms", templatetransform_files)

# --- Step 3: fetch ---
def fetch(repo_name, missing_files):
    repo_dir = f"{repo_base}/{repo_name}"
    if not missing_files:
        print(f"{repo_name}: nothing to get")
        return
    result = subprocess.run(
        ["datalad", "get"] + missing_files,
        capture_output=True, text=True, cwd=repo_dir
    )
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"{repo_name}: ERROR — {result.stderr[-2000:]}")

fetch("studyforrest-data-phase2", missing_phase2)
fetch("studyforrest-data-templatetransforms", missing_tt)
