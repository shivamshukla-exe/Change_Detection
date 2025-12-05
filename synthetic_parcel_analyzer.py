"""
Synthetic Land Parcel Analyzer
Creates parcels around detected buildings using adaptive buffer formula
"""

import cv2
import numpy as np
import rasterio
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os

from rooftop_model_loader import RooftopModelLoader

class SyntheticParcelAnalyzer:
    def __init__(self, model_path="rooftop_best_model_new.pt"):
        """Initialize with rooftop detection model"""
        print("🏗️  Initializing Synthetic Parcel Analyzer")
        self.rooftop_model = RooftopModelLoader(model_path)
        self.buildings = []
        self.parcels = []
        
    def load_image(self, image_path):
        """Load satellite image"""
        print(f"\n📸 Loading image: {image_path}")
        
        with rasterio.open(image_path) as src:
            if src.count >= 3:
                self.image = np.dstack([src.read(1), src.read(2), src.read(3)])
            else:
                band = src.read(1)
                self.image = np.dstack([band, band, band])
            
            self.image = np.clip(self.image, 0, 255).astype(np.uint8)
            
            # Get geospatial info
            self.bounds = src.bounds
            self.resolution = src.res
            
            # Calculate meters per pixel
            lat = (src.bounds.top + src.bounds.bottom) / 2
            self.meters_per_pixel_x = abs(self.resolution[0]) * 111320 * np.cos(np.radians(lat))
            self.meters_per_pixel_y = abs(self.resolution[1]) * 111320
            self.pixel_area_m2 = self.meters_per_pixel_x * self.meters_per_pixel_y
            
            print(f"   Resolution: {self.meters_per_pixel_x:.2f}m x {self.meters_per_pixel_y:.2f}m per pixel")
            print(f"   Image size: {self.image.shape[1]}x{self.image.shape[0]} pixels")
    
    def detect_buildings(self, threshold=0.05, min_area_pixels=100):
        """Detect all buildings in the image"""
        print(f"\n🏠 Detecting buildings...")
        
        # Run rooftop detection
        result = self.rooftop_model.predict(self.image, threshold=threshold)
        rooftop_mask = result['mask']
        
        # Find contours
        contours, _ = cv2.findContours(rooftop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"   Found {len(contours)} potential buildings")
        
        # Process each building
        for contour in contours:
            area_pixels = cv2.contourArea(contour)
            
            if area_pixels < min_area_pixels:
                continue
            
            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rect)
            
            # Get width and height in pixels
            width_px = rect[1][0]
            height_px = rect[1][1]
            
            # Convert to meters
            width_m = width_px * self.meters_per_pixel_x
            height_m = height_px * self.meters_per_pixel_y
            area_m2 = area_pixels * self.pixel_area_m2
            
            # Get center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(rect[0][0]), int(rect[0][1])
            
            self.buildings.append({
                'contour': contour,
                'center': (cx, cy),
                'width_m': width_m,
                'height_m': height_m,
                'area_m2': area_m2,
                'area_pixels': area_pixels,
                'bbox': box_points
            })
        
        print(f"   ✓ Detected {len(self.buildings)} valid buildings")
    
    def calculate_adaptive_c(self, area_m2):
        """
        Calculate adaptive buffer constant based on building size
        
        Small buildings (< 100 m²): c = 0.4 (need more surrounding space)
        Medium buildings (100-500 m²): c = 0.3 (normal)
        Large buildings (500-2000 m²): c = 0.25 (less relative space)
        Very large (> 2000 m²): c = 0.2 (industrial/commercial)
        """
        if area_m2 < 100:
            return 0.4
        elif area_m2 < 500:
            # Linear interpolation between 0.4 and 0.3
            return 0.4 - (area_m2 - 100) * (0.1 / 400)
        elif area_m2 < 2000:
            # Linear interpolation between 0.3 and 0.25
            return 0.3 - (area_m2 - 500) * (0.05 / 1500)
        else:
            # Large buildings get smaller relative buffer
            return max(0.15, 0.25 - (area_m2 - 2000) * 0.00002)
    
    def create_synthetic_parcels(self):
        """Create synthetic land parcels around each building"""
        print(f"\n📐 Creating synthetic parcels...")
        
        for i, building in enumerate(self.buildings):
            # Extract building properties
            Ab = building['area_m2']
            W = building['width_m']
            H = building['height_m']
            cx, cy = building['center']
            
            # Calculate characteristic size
            S = np.sqrt(Ab)
            
            # Get adaptive c
            c = self.calculate_adaptive_c(Ab)
            
            # Calculate buffer distance
            d = c * S
            
            # Calculate parcel area (formula from your spec)
            Ap = (W + 2*d) * (H + 2*d)
            
            # Alternative: expanded formula
            # Ap = Ab + 2*d*(W + H) + 4*d**2
            
            # Calculate unused area
            A_unused = Ap - Ab
            unused_percentage = (A_unused / Ap) * 100
            
            # Convert buffer to pixels
            d_px_x = d / self.meters_per_pixel_x
            d_px_y = d / self.meters_per_pixel_y
            
            # Create parcel rectangle in pixel coordinates
            # Get building bounding box
            x, y, w, h = cv2.boundingRect(building['contour'])
            
            # Expand by buffer
            parcel_x1 = max(0, x - d_px_x)
            parcel_y1 = max(0, y - d_px_y)
            parcel_x2 = min(self.image.shape[1], x + w + d_px_x)
            parcel_y2 = min(self.image.shape[0], y + h + d_px_y)
            
            # Store parcel info
            self.parcels.append({
                'building_id': i,
                'building_area_m2': Ab,
                'parcel_area_m2': Ap,
                'unused_area_m2': A_unused,
                'unused_percentage': unused_percentage,
                'buffer_distance_m': d,
                'c_value': c,
                'bbox_pixels': (parcel_x1, parcel_y1, parcel_x2, parcel_y2),
                'center': (cx, cy),
                'flagged': unused_percentage > 30
            })
        
        # Calculate statistics
        flagged_count = sum(1 for p in self.parcels if p['flagged'])
        print(f"   ✓ Created {len(self.parcels)} synthetic parcels")
        print(f"   ⚠️  {flagged_count} parcels have >30% unused area ({flagged_count/len(self.parcels)*100:.1f}%)")
    
    def visualize_results(self, output_path="synthetic_parcels_result.png"):
        """Create comprehensive visualization"""
        print(f"\n🎨 Creating visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Overview with all parcels
        ax1 = axes[0]
        ax1.imshow(self.image)
        ax1.set_title('All Detected Buildings with Synthetic Parcels', 
                     fontsize=14, fontweight='bold')
        
        for i, (building, parcel) in enumerate(zip(self.buildings, self.parcels)):
            # Draw parcel boundary (yellow rectangle)
            x1, y1, x2, y2 = parcel['bbox_pixels']
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           fill=False, edgecolor='yellow', linewidth=2, alpha=0.8)
            ax1.add_patch(rect)
            
            # Draw building (red circle)
            cx, cy = building['center']
            radius = np.sqrt(building['area_pixels'] / np.pi)
            circle = Circle((cx, cy), radius, 
                          fill=True, facecolor='red', edgecolor='cyan', 
                          linewidth=2, alpha=0.4)
            ax1.add_patch(circle)
            
            # Add label
            color = 'red' if parcel['flagged'] else 'white'
            ax1.text(cx, cy, str(i+1), color=color, fontsize=8, 
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        ax1.axis('off')
        
        # Right: Detailed view of flagged parcels
        ax2 = axes[1]
        ax2.imshow(self.image)
        ax2.set_title('Parcels with >30% Unused Area (Flagged)', 
                     fontsize=14, fontweight='bold', color='red')
        
        for i, (building, parcel) in enumerate(zip(self.buildings, self.parcels)):
            if not parcel['flagged']:
                continue
            
            # Draw parcel
            x1, y1, x2, y2 = parcel['bbox_pixels']
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           fill=True, facecolor='orange', edgecolor='red', 
                           linewidth=3, alpha=0.3)
            ax2.add_patch(rect)
            
            # Draw building
            cx, cy = building['center']
            radius = np.sqrt(building['area_pixels'] / np.pi)
            circle = Circle((cx, cy), radius,
                          fill=True, facecolor='red', edgecolor='yellow',
                          linewidth=2, alpha=0.6)
            ax2.add_patch(circle)
            
            # Add detailed label
            label = f"{i+1}\n{parcel['unused_percentage']:.0f}%"
            ax2.text(cx, cy, label, color='white', fontsize=9,
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.9))
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"   ✓ Saved: {output_path}")
        plt.close()
    
    def create_detailed_report(self, output_path="synthetic_parcel_report.txt"):
        """Generate detailed text report"""
        print(f"\n📄 Creating detailed report...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SYNTHETIC LAND PARCEL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("Methodology:\n")
            f.write("  - Detected buildings using DeepLabV3Plus rooftop model\n")
            f.write("  - Created synthetic parcels using adaptive buffer formula:\n")
            f.write("    • S = √(Building_Area)\n")
            f.write("    • d = c × S  (where c adapts based on building size)\n")
            f.write("    • Parcel_Area = (W + 2d) × (H + 2d)\n")
            f.write("  - Flagged parcels with >30% unused area\n\n")
            
            f.write(f"Summary:\n")
            f.write(f"  Total Buildings Detected: {len(self.buildings)}\n")
            f.write(f"  Total Parcels Created: {len(self.parcels)}\n")
            
            total_building_area = sum(p['building_area_m2'] for p in self.parcels)
            total_parcel_area = sum(p['parcel_area_m2'] for p in self.parcels)
            total_unused = sum(p['unused_area_m2'] for p in self.parcels)
            
            f.write(f"  Total Building Area: {total_building_area:.2f} m² ({total_building_area/10000:.4f} hectares)\n")
            f.write(f"  Total Parcel Area: {total_parcel_area:.2f} m² ({total_parcel_area/10000:.4f} hectares)\n")
            f.write(f"  Total Unused Area: {total_unused:.2f} m² ({total_unused/10000:.4f} hectares)\n")
            f.write(f"  Overall Unused: {(total_unused/total_parcel_area)*100:.2f}%\n\n")
            
            flagged = [p for p in self.parcels if p['flagged']]
            f.write(f"  Flagged Parcels (>30% unused): {len(flagged)} ({len(flagged)/len(self.parcels)*100:.1f}%)\n\n")
            
            f.write("="*80 + "\n")
            f.write("INDIVIDUAL PARCEL DETAILS\n")
            f.write("="*80 + "\n\n")
            
            for i, parcel in enumerate(self.parcels):
                f.write(f"Building #{i+1}:\n")
                f.write(f"  Building Area: {parcel['building_area_m2']:.2f} m²\n")
                f.write(f"  Parcel Area: {parcel['parcel_area_m2']:.2f} m²\n")
                f.write(f"  Unused Area: {parcel['unused_area_m2']:.2f} m²\n")
                f.write(f"  Unused Percentage: {parcel['unused_percentage']:.2f}%\n")
                f.write(f"  Buffer Distance: {parcel['buffer_distance_m']:.2f} m\n")
                f.write(f"  Adaptive c value: {parcel['c_value']:.3f}\n")
                f.write(f"  Status: {'⚠️ FLAGGED (>30% unused)' if parcel['flagged'] else '✓ Normal'}\n")
                f.write("\n")
        
        print(f"   ✓ Saved: {output_path}")
    
    def create_summary_chart(self, output_path="parcel_summary_chart.png"):
        """Create summary charts"""
        print(f"\n📊 Creating summary charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Chart 1: Building size distribution
        ax1 = axes[0, 0]
        areas = [p['building_area_m2'] for p in self.parcels]
        ax1.hist(areas, bins=20, color='skyblue', edgecolor='black')
        ax1.set_title('Building Size Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Building Area (m²)')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Unused percentage distribution
        ax2 = axes[0, 1]
        unused_pcts = [p['unused_percentage'] for p in self.parcels]
        ax2.hist(unused_pcts, bins=20, color='lightcoral', edgecolor='black')
        ax2.axvline(x=30, color='red', linestyle='--', linewidth=2, label='30% threshold')
        ax2.set_title('Unused Area Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Unused Percentage (%)')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Adaptive c values
        ax3 = axes[1, 0]
        c_values = [p['c_value'] for p in self.parcels]
        ax3.scatter(areas, c_values, alpha=0.6, c='green', s=50)
        ax3.set_title('Adaptive c Value vs Building Size', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Building Area (m²)')
        ax3.set_ylabel('c value')
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Parcel vs Building area
        ax4 = axes[1, 1]
        parcel_areas = [p['parcel_area_m2'] for p in self.parcels]
        ax4.scatter(areas, parcel_areas, alpha=0.6, c=unused_pcts, cmap='RdYlGn_r', s=50)
        ax4.plot([0, max(areas)], [0, max(areas)], 'k--', alpha=0.3, label='1:1 line')
        ax4.set_title('Parcel Area vs Building Area', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Building Area (m²)')
        ax4.set_ylabel('Parcel Area (m²)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Unused %')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {output_path}")
        plt.close()

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python synthetic_parcel_analyzer.py <image_path>")
        print("\nExample:")
        print('  python synthetic_parcel_analyzer.py "change detection/2024.tif"')
        return
    
    image_path = sys.argv[1]
    
    print("="*80)
    print("SYNTHETIC LAND PARCEL ANALYZER")
    print("="*80)
    
    # Initialize
    analyzer = SyntheticParcelAnalyzer()
    
    # Load image
    analyzer.load_image(image_path)
    
    # Detect buildings
    analyzer.detect_buildings(threshold=0.05, min_area_pixels=200)
    
    if len(analyzer.buildings) == 0:
        print("\n⚠️  No buildings detected!")
        return
    
    # Create synthetic parcels
    analyzer.create_synthetic_parcels()
    
    # Generate outputs
    analyzer.visualize_results()
    analyzer.create_detailed_report()
    analyzer.create_summary_chart()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  📸 synthetic_parcels_result.png - Visual results")
    print("  📄 synthetic_parcel_report.txt - Detailed report")
    print("  📊 parcel_summary_chart.png - Statistical charts")

if __name__ == "__main__":
    main()
