from datetime import datetime
from pathlib import Path

 # Create new folders for each day, recording interval and object class
def setup_directories(labels, save_raw, save_overlay):
    rec_start = datetime.now().strftime("%Y%m%d_%H-%M")
    save_path = f"insect-detect/data/{rec_start[:8]}/{rec_start}"
    for text in labels:
        Path(f"{save_path}/cropped/{text}").mkdir(parents=True, exist_ok=True)
    if save_raw:
        Path(f"{save_path}/raw").mkdir(parents=True, exist_ok=True)
    if save_overlay:
        Path(f"{save_path}/overlay").mkdir(parents=True, exist_ok=True)
    
    # Calculate current recording ID by subtracting number of directories with date-prefix
    folders_dates = len([f for f in Path("insect-detect/data").glob("**/20*") if f.is_dir()])
    folders_days = len([f for f in Path("insect-detect/data").glob("20*") if f.is_dir()])
    rec_id = folders_dates - folders_days
    
    return save_path, rec_id, rec_start