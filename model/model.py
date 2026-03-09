from nilearn import plotting
import matplotlib.pyplot as plt

# Update this path to exactly match where your new .nii.gz file is located
fmri_file_path = r"F:\Dyslexia\model\datasets\ds003126-download\sub-047EPKL014067\ses-2\func\sub-047EPKL014067_ses-2_task-read_run-01_bold.nii.gz"

# This will plot the structural brain image 
print("Loading fMRI data...")
plotting.plot_epi(fmri_file_path, title="My First fMRI Scan - Dyslexia Project")

# Display the image on your screen
plt.show()