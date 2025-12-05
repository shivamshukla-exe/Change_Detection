# Change Detection Project

Detect changes between before/after satellite or aerial images. Identifies new buildings, encroachments, and structural changes.

## Quick Start (4 hours to submit!)

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Detection

**Simple Command Line:**
```bash
python detect.py "change detection/2024.tif" "change detection/2025.tif"
```

**With Custom Parameters:**
```bash
python detect.py before.jpg after.jpg --threshold 25 --min-area 600 --output result.png
```

**GUI Version (Interactive):**
```bash
python change_detection_gui.py
```

## Parameters

- **threshold** (10-60): Lower = more sensitive, Higher = only major changes
  - Use 15-20 for subtle changes
  - Use 25-35 for significant changes only
  
- **min-area** (100-2000): Minimum size of detected change in pixels
  - Use 300-500 for small buildings
  - Use 800-1500 for large structures only

## Supported Formats

- GeoTIFF (.tif, .tiff) - Satellite imagery
- PNG, JPG, JPEG - Regular photos

## Your TIF Files

Your 2024.tif and 2025.tif files are **NOT corrupted**! They're GeoTIFF satellite images that need special software. Regular photo viewers can't open them, but this tool handles them perfectly.

## Files

- `detect.py` - Main command-line tool (RECOMMENDED)
- `change_detection_gui.py` - Interactive GUI with sliders
- `change_detection_advanced.py` - Advanced version with multiple methods
- `check_tif.py` - Verify TIF files and create previews

## How It Works

1. **Alignment** - Matches features between images to align them
2. **Color Normalization** - Adjusts for lighting/seasonal differences
3. **Edge Detection** - Focuses on structural changes (buildings, roads)
4. **Filtering** - Removes noise and small irrelevant changes
5. **Visualization** - Highlights detected changes with numbered regions

## Tips for Best Results

- Images should be of the same area
- Higher resolution = better detection
- Adjust threshold if too many/few changes detected
- Increase min-area to ignore small changes
- Use GUI to experiment with parameters

## Example Output

The tool creates a 4-panel visualization:
- Top Left: BEFORE image
- Top Right: AFTER image
- Bottom Left: Difference heatmap
- Bottom Right: Detected changes highlighted in red/yellow with numbers

## Quick Parameter Guide

**Too many false positives?**
- Increase threshold (try 30-40)
- Increase min-area (try 1000-1500)

**Missing real changes?**
- Decrease threshold (try 15-20)
- Decrease min-area (try 300-400)

**For your 2024/2025 images:**
```bash
python detect.py "change detection/2024.tif" "change detection/2025.tif" --threshold 22 --min-area 650
```

Good luck with your submission!
