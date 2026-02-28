"""
mid_fmri_analysis.py
====================
Skeleton pipeline for analyzing the Monetary Incentive Delay (MID) task
from fMRIPrep-preprocessed data.

Goal: Isolate ventral striatum (VS) signal and test whether a parametric
regressor tracking trial number (i.e. habituation) explains variance in
the anticipation period, separately for each cue type (win / lose / neutral).

Pipeline overview
-----------------
1. Locate fMRIPrep outputs and events files for a given subject
2. Compute and log motion metrics to a shared exclusion CSV
3. Load and prepare confound regressors (6 HMP + WM + CSF)
4. Build the GLM design matrix:
      - Cue onset regressors  (win, lose, neutral)
      - Anticipation regressors (win, lose, neutral)
          └─ with parametric modulator: z-scored trial number
      - Feedback regressors   (win, lose, neutral)
5. Fit a first-level GLM (nilearn) per run, then combine runs via
   fixed-effects model
6. Extract VS ROI signal using the Harvard-Oxford subcortical atlas
7. Save outputs:
      - Run-level + combined contrast/beta images (.nii.gz)
      - Mean VS timeseries per run (.csv)
      - VS ROI beta values from contrasts of interest (.csv)

Usage
-----
Run a single subject:
    python mid_fmri_analysis.py --subject sub-01

Run all subjects found in the fMRIPrep directory:
    python mid_fmri_analysis.py --all

Dependencies: nilearn, nibabel, numpy, pandas, scipy
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, masking, plotting
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

# ---------------------------------------------------------------------------
# ── CONFIGURATION ──────────────────────────────────────────────────────────
# All paths and analysis parameters live here. Edit before running.
# ---------------------------------------------------------------------------

CONFIG = {
    # --- Directories ---
    "bids_dir":       Path("/projects/b1108/studies/crest/data/raw/neuroimaging/bids"),          # raw BIDS root
    "fmriprep_dir":   Path("/projects/b1108/studies/crest/data/processed/neuroimaging/fmriprep_23.2.0"),      # fMRIPrep derivatives root
    "output_dir":     Path("/projects/b1108/studies/crest/data/processed/neuroimaging/kat_masters"),         # where results will be saved

    # --- fMRIPrep filename identifiers ---
    "space":          "space-MNI152NLin6Asym",
    "resolution":     "res-2",                         # adjust if needed (e.g. "res-2")
    "task":           "MID",
    "n_runs":         2,

    # --- GLM parameters ---
    "t_r":            2.05,          # repetition time in seconds
    "hrf_model":      "spm",        # HRF model passed to nilearn
    "drift_model":    "cosine",
    "high_pass":      0.01,         # Hz — applied via drift model
    "smoothing_fwhm": 6,            # mm; set None to skip smoothing in GLM

    # --- MID trial structure ---
    # Column names expected in events.tsv — adjust to match your file
    "trial_type_col": "trial_type",   # column that encodes condition labels
    "cue_types":      ["win", "lose", "neutral"],   # values in trial_type_col
    # Expected trial_type labels — the script will look for strings containing
    # these keywords, so "win_cue", "win_anticipation", "win_feedback" all work
    "cue_label":          "cue",
    "anticipation_label": "anticipation",
    "feedback_label":     "feedback",

    # --- Confound regressor column names (from fMRIPrep confounds TSV) ---
    "confound_cols": [
        "trans_x", "trans_y", "trans_z",
        "rot_x",   "rot_y",   "rot_z",
        "white_matter", "csf",
    ],

    # --- Motion exclusion thresholds ---
    "fd_threshold":          0.5,    # mm — volumes above this are "high motion"
    "fd_col":                "framewise_displacement",
    "pct_highmotion_thresh": 20.0,   # % of volumes; flag run if exceeded

    # --- VS ROI (Harvard-Oxford subcortical atlas) ---
    # Label indices for left and right nucleus accumbens / ventral striatum.
    # Run `fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")` and inspect
    # .labels to confirm the correct indices for your version of the atlas.
    #TODO this is incorrect
    "vs_label_indices": [11, 21],    # example: L/R nucleus accumbens area
    print(fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm".labels)

    # --- Output CSV for motion tracking ---
    "motion_csv": Path("/projects/b1108/studies/crest/data/processed/neuroimaging/kat_masters.csv"),
}

# Contrasts of interest (name: contrast vector will be built automatically)
# These names correspond to regressors that will exist in the design matrix.
CONTRASTS = {
    # Mean activation (still useful as a sanity check / baseline characterization)
    "win_anticipation":     "win_anticipation",
    "lose_anticipation":    "lose_anticipation",
    "neutral_anticipation": "neutral_anticipation",

    # These are your actual question — does VS response habituate over trials?
    "win_trial_num":        "win_anticipation_x_trial_num",
    "lose_trial_num":       "lose_anticipation_x_trial_num",
    "neutral_trial_num":    "neutral_anticipation_x_trial_num",
}

# ---------------------------------------------------------------------------
# ── PATH HELPERS ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def get_bold_path(subject: str, run: int) -> Path:
    """
    Return the path to a subject's preprocessed BOLD image for a given run.
    Constructs the standard fMRIPrep output filename.
    """
    fname = (
        f"{subject}_task-{CONFIG['task']}_run-{run:02d}"
        f"_{CONFIG['resolution']}_{CONFIG['space']}_desc-preproc_bold.nii.gz"
    )
    return CONFIG["fmriprep_dir"] / subject / "func" / fname


def get_confounds_path(subject: str, run: int) -> Path:
    """Return path to fMRIPrep confounds TSV for a given subject and run."""
    fname = (
        f"{subject}_task-{CONFIG['task']}_run-{run:02d}"
        f"_desc-confounds_timeseries.tsv"
    )
    return CONFIG["fmriprep_dir"] / subject / "func" / fname


def get_events_path(subject: str, run: int) -> Path:
    """Return path to BIDS events TSV for a given subject and run."""
    fname = f"{subject}_task-{CONFIG['task']}_run-{run:02d}_events.tsv"
    return CONFIG["bids_dir"] / subject / "func" / fname


def get_output_dir(subject: str) -> Path:
    """Create and return the subject-level output directory."""
    out = CONFIG["output_dir"] / subject
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# ── MOTION QC ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def compute_motion_metrics(confounds_df: pd.DataFrame) -> dict:
    """
    Compute mean framewise displacement and % of high-motion volumes
    from a fMRIPrep confounds dataframe.

    Returns a dict with keys: mean_fd, pct_highmotion, n_volumes_total,
    n_highmotion_volumes.
    """
    fd = confounds_df[CONFIG["fd_col"]].fillna(0)   # first vol is NaN in fMRIPrep
    n_total        = len(fd)
    n_highmotion   = int((fd > CONFIG["fd_threshold"]).sum())
    pct_highmotion = 100 * n_highmotion / n_total

    return {
        "mean_fd":             round(fd.mean(), 4),
        "pct_highmotion":      round(pct_highmotion, 2),
        "n_volumes_total":     n_total,
        "n_highmotion_volumes": n_highmotion,
    }


def update_motion_csv(subject: str, run: int, metrics: dict) -> None:
    """
    Update the shared motion summary CSV with QC metrics for one run.

    Reads the existing CSV if it exists (so previously processed subjects
    are preserved), inserts/updates the row for this subject+run, then
    writes back. This means re-running a single subject safely overwrites
    only their rows.
    """
    csv_path = CONFIG["motion_csv"]

    # Load existing data or start fresh
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    # Build the new row
    row = {
        "subject": subject,
        "run":     run,
        **metrics,
        "flagged": (
            metrics["mean_fd"]        > CONFIG["fd_threshold"] or
            metrics["pct_highmotion"] > CONFIG["pct_highmotion_thresh"]
        ),
    }

    # Drop any existing entry for this subject+run, then append
    if not df.empty and {"subject", "run"}.issubset(df.columns):
        df = df[~((df["subject"] == subject) & (df["run"] == run))]

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["subject", "run"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"  [motion] Updated {csv_path} for {subject} run-{run:02d}")


# ---------------------------------------------------------------------------
# ── CONFOUND PREPARATION ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def load_confounds(confounds_path: Path) -> pd.DataFrame:
    """
    Load fMRIPrep confounds TSV and return only the columns we need
    (6 motion params + WM + CSF), with NaNs filled by column mean.

    Returns a DataFrame ready to pass directly to FirstLevelModel.
    """
    confounds_df = pd.read_csv(confounds_path, sep="\t")

    missing = [c for c in CONFIG["confound_cols"] if c not in confounds_df.columns]
    if missing:
        raise ValueError(f"Confound columns not found in {confounds_path}: {missing}")

    selected = confounds_df[CONFIG["confound_cols"]].copy()
    selected = selected.fillna(selected.mean())   # handles first-row NaN in FD etc.
    return selected


# ---------------------------------------------------------------------------
# ── EVENTS / DESIGN MATRIX ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def load_and_label_events(events_path: Path) -> pd.DataFrame:
    """
    Load an events TSV and return a tidy DataFrame with one row per event,
    columns: onset, duration, trial_type.

    Also adds a 'trial_number' column that numbers trials sequentially
    within each cue type (used as the parametric modulator).
    """
    events = pd.read_csv(events_path, sep="\t")

    # Validate required columns
    required = {"onset", "duration", CONFIG["trial_type_col"]}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events.tsv missing columns: {missing} in {events_path}")

    # Rename to the standard 'trial_type' that nilearn expects
    if CONFIG["trial_type_col"] != "trial_type":
        events = events.rename(columns={CONFIG["trial_type_col"]: "trial_type"})

    events = events.sort_values("onset").reset_index(drop=True)

    # Assign sequential trial number within each cue x phase combination
    # (used downstream to build the parametric modulator)
    events["trial_number"] = (
        events.groupby("trial_type").cumcount() + 1
    )

    return events


def build_parametric_modulator(events: pd.DataFrame, cue_type: str) -> pd.Series:
    """
    For anticipation events of a given cue type, return a z-scored trial
    number series aligned to all events (0 for non-matching rows).

    Z-scoring centers and scales the modulator so it's orthogonal to the
    mean anticipation regressor and comparable across cue types.
    """
    anticipation_label = f"{cue_type}_{CONFIG['anticipation_label']}"
    mask = events["trial_type"] == anticipation_label

    modulator = pd.Series(np.zeros(len(events)), index=events.index)
    trial_nums = events.loc[mask, "trial_number"].astype(float)

    if trial_nums.std() > 0:
        modulator[mask] = (trial_nums - trial_nums.mean()) / trial_nums.std()
    else:
        warnings.warn(f"Trial number SD=0 for {anticipation_label}; modulator will be zeros.")

    return modulator


def build_events_for_glm(events: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the events DataFrame to include parametric modulator columns
    required by nilearn's make_first_level_design_matrix.

    nilearn expects a 'modulation' column when you want parametric regressors.
    We create separate rows for the base anticipation regressor (modulation=1)
    and for the trial-number modulator (modulation=z-scored trial number).

    Returns a DataFrame with columns: onset, duration, trial_type, modulation.
    """
    rows = []

    for _, event in events.iterrows():
        trial_type = event["trial_type"]

        # All events get a standard regressor (modulation = 1)
        rows.append({
            "onset":      event["onset"],
            "duration":   event["duration"],
            "trial_type": trial_type,
            "modulation": 1.0,
        })

        # For anticipation events only: add a second row for the parametric
        # modulator, named  <cue>_anticipation_x_trial_num
        for cue in CONFIG["cue_types"]:
            anticipation_label = f"{cue}_{CONFIG['anticipation_label']}"
            if trial_type == anticipation_label:
                z_trial = build_parametric_modulator(events, cue)[event.name]
                rows.append({
                    "onset":      event["onset"],
                    "duration":   event["duration"],
                    "trial_type": f"{anticipation_label}_x_trial_num",
                    "modulation": z_trial,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ── VS ROI MASK ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def get_vs_mask() -> nib.Nifti1Image:
    """
    Fetch the Harvard-Oxford subcortical atlas and extract a binary mask
    for the ventral striatum (nucleus accumbens) using the label indices
    defined in CONFIG['vs_label_indices'].

    Returns a binary NIfTI mask image in MNI space.
    """
    print("  [ROI] Fetching Harvard-Oxford subcortical atlas...")
    atlas = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    atlas_img = atlas.maps

    atlas_data = atlas_img.get_fdata()
    vs_mask    = np.zeros_like(atlas_data, dtype=np.int8)

    for idx in CONFIG["vs_label_indices"]:
        vs_mask[atlas_data == idx] = 1

    vs_mask_img = nib.Nifti1Image(vs_mask, atlas_img.affine, atlas_img.header)
    print(f"  [ROI] VS mask: {vs_mask.sum()} voxels across label indices {CONFIG['vs_label_indices']}")
    return vs_mask_img


# ---------------------------------------------------------------------------
# ── GLM FITTING ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def fit_run_glm(
    bold_img:     nib.Nifti1Image,
    events_df:    pd.DataFrame,
    confounds_df: pd.DataFrame,
    subject:      str,
    run:          int,
) -> FirstLevelModel:
    """
    Fit a first-level GLM for a single run.

    Parameters
    ----------
    bold_img     : preprocessed BOLD image for this run
    events_df    : events DataFrame from build_events_for_glm()
    confounds_df : confounds DataFrame from load_confounds()
    subject      : subject ID string (for logging)
    run          : run number (for logging)

    Returns the fitted FirstLevelModel.
    """
    print(f"  [GLM] Fitting run-{run:02d} for {subject}...")

    glm = FirstLevelModel(
        t_r=CONFIG["t_r"],
        hrf_model=CONFIG["hrf_model"],
        drift_model=CONFIG["drift_model"],
        high_pass=CONFIG["high_pass"],
        smoothing_fwhm=CONFIG["smoothing_fwhm"],
        standardize=False,
        noise_model="ar1",
        verbose=0,
    )

    glm.fit(bold_img, events=events_df, confounds=confounds_df)
    return glm


def compute_run_contrasts(
    glm:     FirstLevelModel,
    subject: str,
    run:     int,
    out_dir: Path,
) -> dict:
    """
    Compute and save contrast images for a single fitted run GLM.

    Saves z-stat and effect-size maps for each contrast in CONTRASTS.
    Returns a dict mapping contrast name -> effect size NIfTI (for fixed-fx).
    """
    effect_imgs = {}

    for contrast_name, contrast_def in CONTRASTS.items():
        try:
            z_map      = glm.compute_contrast(contrast_def, output_type="z_score")
            effect_map = glm.compute_contrast(contrast_def, output_type="effect_size")
            var_map    = glm.compute_contrast(contrast_def, output_type="effect_variance")

            # Save images
            stem = f"{subject}_run-{run:02d}_{contrast_name}"
            nib.save(z_map,      out_dir / f"{stem}_zstat.nii.gz")
            nib.save(effect_map, out_dir / f"{stem}_effect.nii.gz")

            effect_imgs[contrast_name] = (effect_map, var_map)

        except Exception as e:
            warnings.warn(f"Contrast '{contrast_name}' failed for {subject} run-{run}: {e}")

    return effect_imgs


# ---------------------------------------------------------------------------
# ── FIXED-EFFECTS COMBINATION ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

def combine_runs_fixed_effects(
    run_contrast_imgs: List[Dict],
    subject:           str,
    out_dir:           Path,
) -> None:
    """
    Combine run-level contrast images using a fixed-effects model (nilearn).

    run_contrast_imgs : list (one per run) of dicts {contrast_name: (effect_img, var_img)}
    Saves combined z-stat and effect images per contrast.
    """
    print(f"  [fixed-fx] Combining runs for {subject}...")

    # Collect all contrast names (intersection across runs to be safe)
    contrast_names = set(run_contrast_imgs[0].keys())
    for run_imgs in run_contrast_imgs[1:]:
        contrast_names &= set(run_imgs.keys())

    for contrast_name in contrast_names:
        effect_imgs = [run_imgs[contrast_name][0] for run_imgs in run_contrast_imgs]
        var_imgs    = [run_imgs[contrast_name][1] for run_imgs in run_contrast_imgs]

        combined_effect, combined_var, combined_z = compute_fixed_effects(
            effect_imgs, var_imgs, precision_weighted=True
        )

        stem = f"{subject}_combined_{contrast_name}"
        nib.save(combined_z,      out_dir / f"{stem}_zstat.nii.gz")
        nib.save(combined_effect, out_dir / f"{stem}_effect.nii.gz")


# ---------------------------------------------------------------------------
# ── VS ROI SIGNAL EXTRACTION ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

def extract_vs_timeseries(
    bold_img:   nib.Nifti1Image,
    vs_mask:    nib.Nifti1Image,
    subject:    str,
    run:        int,
    out_dir:    Path,
) -> pd.DataFrame:
    """
    Extract mean BOLD signal within the VS mask for each volume.

    Saves a CSV with columns: volume_index, mean_vs_signal.
    Returns the DataFrame.
    """
    masker = NiftiMasker(
        mask_img=vs_mask,
        standardize=False,
        detrend=False,      # detrending handled in GLM
        verbose=0,
    )
    vs_signals = masker.fit_transform(bold_img)   # shape: (n_volumes, n_voxels)
    mean_signal = vs_signals.mean(axis=1)

    df = pd.DataFrame({
        "volume_index":    np.arange(len(mean_signal)),
        "mean_vs_signal":  mean_signal,
    })

    fname = out_dir / f"{subject}_run-{run:02d}_VS_timeseries.csv"
    df.to_csv(fname, index=False)
    print(f"  [ROI] Saved VS timeseries → {fname}")
    return df


def extract_vs_betas(
    contrast_imgs: dict,
    vs_mask:       nib.Nifti1Image,
    subject:       str,
    run_label:     str,
    out_dir:       Path,
) -> None:
    """
    Extract mean VS ROI beta/effect values from contrast images and save to CSV.

    contrast_imgs : dict {contrast_name: (effect_img, var_img)}
    run_label     : e.g. "run-01" or "combined"
    """
    rows = []
    for contrast_name, (effect_img, _) in contrast_imgs.items():
        mean_beta = masking.apply_mask(effect_img, vs_mask).mean()
        rows.append({"subject": subject, "run": run_label,
                     "contrast": contrast_name, "mean_vs_beta": round(float(mean_beta), 6)})

    df = pd.DataFrame(rows)
    fname = out_dir / f"{subject}_{run_label}_VS_betas.csv"
    df.to_csv(fname, index=False)
    print(f"  [ROI] Saved VS betas → {fname}")


# ---------------------------------------------------------------------------
# ── SUBJECT-LEVEL PIPELINE ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def process_subject(subject: str, vs_mask: nib.Nifti1Image) -> None:
    """
    Run the full pipeline for one subject:
        1. Loop over runs: load data, QC motion, fit GLM, extract ROI signal
        2. Combine runs via fixed-effects model
        3. Extract VS betas from combined map
    """
    print(f"\n{'='*60}")
    print(f"Processing {subject}")
    print(f"{'='*60}")

    out_dir = get_output_dir(subject)
    run_contrast_imgs   = []   # accumulate per-run contrast images for fixed-fx
    run_glms            = []   # store fitted GLMs if needed for inspection

    for run in range(1, CONFIG["n_runs"] + 1):

        # --- Locate files ---
        bold_path      = get_bold_path(subject, run)
        confounds_path = get_confounds_path(subject, run)
        events_path    = get_events_path(subject, run)

        for p in [bold_path, confounds_path, events_path]:
            if not p.exists():
                raise FileNotFoundError(f"Expected file not found: {p}")

        print(f"\n  Run {run}")

        # --- Load data ---
        bold_img     = image.load_img(str(bold_path))
        confounds_df = load_confounds(confounds_path)
        events_raw   = load_and_label_events(events_path)

        # --- Motion QC (update shared CSV) ---
        motion_confounds_df = pd.read_csv(confounds_path, sep="\t")
        motion_metrics      = compute_motion_metrics(motion_confounds_df)
        update_motion_csv(subject, run, motion_metrics)
        print(f"  [motion] mean FD={motion_metrics['mean_fd']} mm | "
              f"{motion_metrics['pct_highmotion']}% high-motion volumes")

        # --- Build design matrix events ---
        events_for_glm = build_events_for_glm(events_raw)

        # --- Fit GLM ---
        glm = fit_run_glm(bold_img, events_for_glm, confounds_df, subject, run)
        run_glms.append(glm)

        # --- Compute and save run-level contrasts ---
        contrast_imgs = compute_run_contrasts(glm, subject, run, out_dir)
        run_contrast_imgs.append(contrast_imgs)

        # --- Extract VS timeseries ---
        extract_vs_timeseries(bold_img, vs_mask, subject, run, out_dir)

        # --- Extract VS betas for this run ---
        extract_vs_betas(contrast_imgs, vs_mask, subject, f"run-{run:02d}", out_dir)

    # --- Fixed-effects combination across runs ---
    if len(run_contrast_imgs) == CONFIG["n_runs"]:
        combine_runs_fixed_effects(run_contrast_imgs, subject, out_dir)

        # Load combined effect images back to extract VS betas
        # (build a minimal dict from saved files for reuse of extract_vs_betas)
        combined_contrast_imgs = {}
        for contrast_name in CONTRASTS:
            stem        = out_dir / f"{subject}_combined_{contrast_name}"
            effect_path = stem.with_name(stem.name + "_effect.nii.gz")
            var_path    = stem.with_name(stem.name + "_zstat.nii.gz")   # used as placeholder
            if effect_path.exists():
                combined_contrast_imgs[contrast_name] = (
                    image.load_img(str(effect_path)),
                    image.load_img(str(effect_path)),   # var not needed here
                )
        extract_vs_betas(combined_contrast_imgs, vs_mask, subject, "combined", out_dir)
    else:
        warnings.warn(f"Expected {CONFIG['n_runs']} runs but only got "
                      f"{len(run_contrast_imgs)} for {subject}. Skipping fixed-fx.")

    print(f"\n  ✓ {subject} complete. Outputs in {out_dir}")


# ---------------------------------------------------------------------------
# ── ENTRY POINT ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def get_all_subjects() -> list[str]:
    """Return sorted list of subject IDs found in the fMRIPrep directory."""
    fmriprep_dir = CONFIG["fmriprep_dir"]
    subjects = sorted([
        d.name for d in fmriprep_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])
    return subjects


def main():
    parser = argparse.ArgumentParser(
        description="MID fMRI pipeline: VS signal extraction and habituation analysis."
    )
    parser.add_argument("--subject",  type=str, help="Single subject ID, e.g. sub-01")
    parser.add_argument("--all",      action="store_true", help="Process all subjects")
    args = parser.parse_args()

    if not args.subject and not args.all:
        parser.error("Provide --subject <id> or --all")

    # Build VS mask once (shared across all subjects)
    vs_mask = get_vs_mask()

    if args.all:
        subjects = get_all_subjects()
        print(f"Found {len(subjects)} subjects: {subjects}")
    else:
        subjects = [args.subject]

    for subject in subjects:
        try:
            process_subject(subject, vs_mask)
        except Exception as e:
            warnings.warn(f"Subject {subject} failed: {e}")
            continue

    print("\nAll done.")


if __name__ == "__main__":
    main()