Author: Tamar Japaridze
Document: ROI_analysis.ipynb

The overall aim of this project was to perform a classification analysis for the functionally defined Extrastriate Body Area, to see whether
its voxels differentially represent positive and negative-valenced actions as seen in a naturalistic audiovisual stimulus (the Forrest Gump movie).

To accomplish this, here are the steps I successfully took (with help from my groupmates and AI):
- Locate the most preprocessed versions of BOLD data for this dataset
- Locate the functional ROI masks for this dataset
- Double-checked that the dimensions of these nifti files are aligned
- Extracted events (pos, neg body contacts) and their timings from the annotated body contact list
- Converted from event timings to TRs across BOLD runs
- Saved the related BOLD responses for each event and their labels (positive or negative)

Steps still left:
- Running the actual classification analysis in a cross-validated manner
- Communicating results
- Exploring other analyses from this rich but disorganized dataset
