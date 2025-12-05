"""
Building Change Detection using Trained Rooftop Segmentation Model
Detects NEW buildings by comparing rooftop masks from BEFORE and AFTER images
"""

import torch
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
import sys
import os

# Import helper modules
from rooftop_model_loader import RooftopModelLoader

class RooftopChangeDetector:
    def __init__(self, model_path="rooftop_best_model_new.pt", device='cpu'):
        """Initialize with trained rooftop model"""
        print("="*70)
        print("ROOFTOP-BASED BUILDING CHANGE DETECTION")
        print("="*70)
        
        self.device = device
        self.model_loader = RooftopModelLoader(model_path, device)
        
    def load_image(self, filepath):
        """Load image - supports GeoTIFF, PNG, JPG"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.tif', '.tiff']:
            with rasterio.open(filepath) as src:
                # Store geospatial metadata
                self.crs = src.crs
                self.transform = src.transform
                self.bounds = src.bounds
                self.resolution = src.res  # (x_res, y_res) in degrees
                
                # Calculate meters per pixel at this latitude
                # At latitude ~23°, 1 degree ≈ 111 km
                lat = (src.bounds.top + src.bounds.bottom) / 2
                meters_per_degree_lat = 111320  # meters
                meters_per_degree_lon = 111320 * np.cos(np.radians(lat))
                
                self.meters_per_pixel_x = abs(self.resolution[0]) * meters_per_degree_lon
                self.meters_per_pixel_y = abs(self.resolution[1]) * meters_per_degree_lat
                self.pixel_area_m2 = self.meters_per_pixel_x * self.meters_per_pixel_y
                
                print(f"   📏 Resolution: {self.meters_per_pixel_x:.2f}m x {self.meters_per_pixel_y:.2f}m per pixel")
                print(f"   📐 Pixel area: {self.pixel_area_m2:.2f} m²")
                
                if src.count >= 3:
                    img = np.dstack([src.read(1), src.read(2), src.read(3)])
                else:
                    img = src.read(1)
                    img = np.dstack([img, img, img])
                return np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = cv2.imread(filepath)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def align_images(self, img1, img2):
        """Align images using simple resize (no warping to avoid artifacts)"""
        print("\n📐 Aligning images...")
        
        # Just resize to match dimensions - simpler and cleaner
        h, w = img1.shape[:2]
        img2_resized = cv2.resize(img2, (w, h))
        
        print(f"   Resized to {w}x{h}")
        return img1, img2_resized
    
    def detect_new_buildings(self, before_path, after_path, threshold=0.05, min_area=300):
        """
        Main detection pipeline using trained rooftop model
        
        Args:
            before_path: Path to BEFORE image
            after_path: Path to AFTER image
            threshold: Rooftop detection threshold (0.05 recommended)
            min_area: Minimum building area in pixels
        """
        print(f"\n📂 Loading images...")
        print(f"   BEFORE: {before_path}")
        print(f"   AFTER:  {after_path}")
        
        # Load images
        before_img = self.load_image(before_path)
        after_img = self.load_image(after_path)
        
        # Align images
        before_aligned, after_aligned = self.align_images(before_img, after_img)
        
        # Detect rooftops in BEFORE image
        print(f"\n🏠 Detecting rooftops in BEFORE image (2024)...")
        before_result = self.model_loader.predict(before_aligned, threshold=threshold)
        before_mask = before_result['mask']
        print(f"   Rooftop pixels: {np.sum(before_mask)}")
        
        # Detect rooftops in AFTER image
        print(f"\n🏠 Detecting rooftops in AFTER image (2025)...")
        after_result = self.model_loader.predict(after_aligned, threshold=threshold)
        after_mask = after_result['mask']
        print(f"   Rooftop pixels: {np.sum(after_mask)}")
        
        # Find NEW rooftops (in AFTER but not in BEFORE)
        print(f"\n🔍 Finding NEW buildings...")
        new_buildings_mask = cv2.subtract(after_mask, before_mask)
        
        # Clean up the mask
        kernel = np.ones((7,7), np.uint8)
        new_buildings_mask = cv2.morphologyEx(new_buildings_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        new_buildings_mask = cv2.morphologyEx(new_buildings_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours of new buildings
        contours, _ = cv2.findContours(new_buildings_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area and shape
        new_buildings = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                # Check if it's somewhat rectangular (buildings are)
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = max(w, h) / (min(w, h) + 1)
                
                # Buildings shouldn't be too elongated
                if aspect_ratio < 10:
                    # Check solidity (how "filled" the shape is)
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        # Buildings are fairly solid shapes
                        if solidity > 0.4:
                            new_buildings.append(c)
        
        print(f"\n{'='*70}")
        print(f"✅ FOUND {len(new_buildings)} NEW BUILDINGS!")
        print(f"{'='*70}")
        
        return {
            'before_img': before_aligned,
            'after_img': after_aligned,
            'before_mask': before_mask,
            'after_mask': after_mask,
            'new_mask': new_buildings_mask,
            'new_buildings': new_buildings
        }
    
    def visualize_results(self, results, output_path="rooftop_change_detection_result.png"):
        """Create comprehensive visualization"""
        before_img = results['before_img']
        after_img = results['after_img']
        before_mask = results['before_mask']
        after_mask = results['after_mask']
        new_mask = results['new_mask']
        new_buildings = results['new_buildings']
        
        # Create result image with new buildings highlighted using CIRCLES
        result_img = after_img.copy()
        has_geo = hasattr(self, 'pixel_area_m2')
        
        for i, contour in enumerate(new_buildings):
            # Get center and calculate appropriate radius
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate radius based on contour area
                area_pixels = cv2.contourArea(contour)
                radius = int(np.sqrt(area_pixels / np.pi) * 1.2)  # 20% larger than actual
                radius = max(30, min(radius, 80))  # Clamp between 30-80 pixels
                
                # Draw semi-transparent red circle (more visible)
                overlay = result_img.copy()
                cv2.circle(overlay, (cx, cy), radius, (255, 50, 50), -1)
                result_img = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)
                
                # Draw bright yellow circle outline (thicker)
                cv2.circle(result_img, (cx, cy), radius, (0, 255, 255), 5)
                
                # Draw white circle for number label
                label_radius = 22
                cv2.circle(result_img, (cx, cy), label_radius, (255, 255, 255), -1)
                cv2.circle(result_img, (cx, cy), label_radius, (0, 0, 0), 2)
                
                # Add number
                text = f"{i+1}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = cx - text_size[0] // 2
                text_y = cy + text_size[1] // 2
                cv2.putText(result_img, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Add area label below the circle
                if has_geo:
                    area_m2 = area_pixels * self.pixel_area_m2
                    area_text = f"{area_m2:.0f}m²"
                    
                    # Draw background for area text
                    text_size = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = cx - text_size[0] // 2
                    text_y = cy + radius + 20
                    
                    # Black background
                    cv2.rectangle(result_img, 
                                (text_x - 3, text_y - text_size[1] - 3),
                                (text_x + text_size[0] + 3, text_y + 3),
                                (0, 0, 0), -1)
                    
                    # White text
                    cv2.putText(result_img, area_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Create visualization with better layout
        fig = plt.figure(figsize=(22, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.1, top=0.95, bottom=0.05, left=0.05, right=0.95)
        
        # Row 1
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(before_img)
        ax1.set_title('BEFORE (2024)', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(after_img)
        ax2.set_title('AFTER (2025)', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(before_mask, cmap='Reds', alpha=0.8)
        ax3.set_title('Rooftops Detected (BEFORE)', fontsize=14, fontweight='bold', pad=10)
        ax3.axis('off')
        
        # Row 2
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(after_mask, cmap='Reds', alpha=0.8)
        ax4.set_title('Rooftops Detected (AFTER)', fontsize=14, fontweight='bold', pad=10)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(new_mask, cmap='hot')
        ax5.set_title('New Buildings Mask', fontsize=14, fontweight='bold', pad=10)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(result_img)
        ax6.set_title(f'NEW BUILDINGS: {len(new_buildings)} detected', 
                     fontsize=14, fontweight='bold', color='red', pad=10)
        ax6.axis('off')
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        print(f"\n💾 Results saved: {output_path}")
        
        # Close the figure to prevent display issues
        plt.close(fig)
        
        return result_img
    
    def save_detailed_report(self, results, output_path="building_change_report.txt"):
        """Save detailed text report with real-world measurements"""
        new_buildings = results['new_buildings']
        
        # Check if we have geospatial metadata
        has_geo = hasattr(self, 'pixel_area_m2')
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BUILDING CHANGE DETECTION REPORT\n")
            f.write("="*70 + "\n\n")
            
            if has_geo:
                f.write(f"Image Resolution: {self.meters_per_pixel_x:.2f}m x {self.meters_per_pixel_y:.2f}m per pixel\n")
                f.write(f"Coordinate System: {self.crs}\n")
                f.write(f"Location: Lat {self.bounds.bottom:.4f}° to {self.bounds.top:.4f}°, ")
                f.write(f"Lon {self.bounds.left:.4f}° to {self.bounds.right:.4f}°\n\n")
            
            f.write(f"Total New Buildings Detected: {len(new_buildings)}\n\n")
            
            total_area_pixels = sum(cv2.contourArea(c) for c in new_buildings)
            if has_geo:
                total_area_m2 = total_area_pixels * self.pixel_area_m2
                f.write(f"Total New Construction Area: {total_area_m2:.2f} m² ({total_area_m2/10000:.4f} hectares)\n\n")
            
            f.write("Building Details:\n")
            f.write("-"*70 + "\n")
            
            for i, contour in enumerate(new_buildings):
                area_pixels = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                f.write(f"\nBuilding #{i+1}:\n")
                f.write(f"  Location (pixel): ({x}, {y})\n")
                
                if has_geo:
                    # Convert pixel coordinates to lat/lon
                    lon = self.bounds.left + x * self.resolution[0]
                    lat = self.bounds.top + y * self.resolution[1]
                    f.write(f"  Location (geo): {lat:.6f}°N, {lon:.6f}°E\n")
                    
                    # Real-world measurements
                    area_m2 = area_pixels * self.pixel_area_m2
                    width_m = w * self.meters_per_pixel_x
                    height_m = h * self.meters_per_pixel_y
                    
                    f.write(f"  Area: {area_m2:.2f} m² ({area_pixels:.0f} pixels)\n")
                    f.write(f"  Dimensions: {width_m:.1f}m x {height_m:.1f}m\n")
                    f.write(f"  Bounding Box: {w}x{h} pixels\n")
                else:
                    f.write(f"  Area: {area_pixels:.0f} pixels\n")
                    f.write(f"  Size: {w}x{h} pixels\n")
        
        print(f"📄 Report saved: {output_path}")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect new buildings using rooftop segmentation')
    parser.add_argument('before', help='Path to BEFORE image')
    parser.add_argument('after', help='Path to AFTER image')
    parser.add_argument('--threshold', type=float, default=0.05, 
                       help='Rooftop detection threshold (default: 0.05)')
    parser.add_argument('--min-area', type=int, default=400, 
                       help='Minimum building area in pixels (default: 400)')
    parser.add_argument('--output', default='rooftop_change_detection_result.png',
                       help='Output image path')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run model on')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RooftopChangeDetector(device=args.device)
    
    # Detect changes
    results = detector.detect_new_buildings(
        args.before,
        args.after,
        threshold=args.threshold,
        min_area=args.min_area
    )
    
    # Visualize
    detector.visualize_results(results, args.output)
    
    # Save report
    detector.save_detailed_report(results)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\n" + "="*70)
        print("ROOFTOP-BASED BUILDING CHANGE DETECTION")
        print("="*70)
        print("\nUsage: python rooftop_change_detection.py <before_image> <after_image> [options]")
        print("\nExample:")
        print('  python rooftop_change_detection.py "change detection/2024.tif" "change detection/2025.tif"')
        print("\nOptions:")
        print("  --threshold   Rooftop detection threshold (default: 0.05)")
        print("  --min-area    Minimum building area (default: 400)")
        print("  --output      Output filename")
        print("  --device      cpu or cuda")
        print("\n" + "="*70)
        sys.exit(1)
    
    main()
