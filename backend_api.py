"""
FastAPI Backend for Building Analysis Suite
This wraps the existing Gradio functions to provide REST API endpoints
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import rasterio
from PIL import Image
import io
import base64
import tempfile
import os
from typing import Optional

from rooftop_change_detection import RooftopChangeDetector
from synthetic_parcel_analyzer import SyntheticParcelAnalyzer

app = FastAPI(title="Building Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path"""
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = upload_file.file.read()
            tmp.write(content)
            tmp.flush()
            return tmp.name
    finally:
        upload_file.file.close()


def image_to_base64(img: np.ndarray) -> str:
    """Convert numpy array image to base64 string"""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def load_image_safe(file_path: str) -> np.ndarray:
    """Safely load image from file path"""
    try:
        with rasterio.open(file_path) as src:
            if src.count >= 3:
                img = np.dstack([src.read(1), src.read(2), src.read(3)])
            else:
                band = src.read(1)
                img = np.dstack([band, band, band])
            return np.clip(img, 0, 255).astype(np.uint8)
    except:
        img = np.array(Image.open(file_path))
        if len(img.shape) == 2:
            img = np.dstack([img, img, img])
        return img


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Building Analysis API is running"}


@app.post("/api/change-detection")
async def change_detection(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    threshold: float = Form(0.05),
    min_area: int = Form(500),
    marker_style: str = Form("Circles with Numbers"),
    show_area: str = Form("true"),
    circle_size: str = Form("Auto")
):
    """
    Detect new buildings by comparing before and after images
    """
    before_path = None
    after_path = None

    try:
        before_path = save_upload_file(before_image)
        after_path = save_upload_file(after_image)

        show_area_bool = show_area.lower() == "true"

        detector = RooftopChangeDetector()
        results = detector.detect_new_buildings(
            before_path,
            after_path,
            threshold=threshold,
            min_area=min_area
        )

        show_numbers = marker_style == "Circles with Numbers"
        # Handle case-insensitive circle_size
        size_mult = {"small": 0.9, "auto": 1.2, "large": 1.5}.get(circle_size.lower(), 1.2)

        result_img = results['after_img'].copy()
        has_geo = hasattr(detector, 'pixel_area_m2') and detector.pixel_area_m2 is not None

        for i, contour in enumerate(results['new_buildings']):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                area_pixels = cv2.contourArea(contour)
                radius = int(np.sqrt(area_pixels / np.pi) * size_mult)
                radius = max(25, min(radius, 100))

                overlay = result_img.copy()
                cv2.circle(overlay, (cx, cy), radius, (255, 50, 50), -1)
                result_img = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)
                cv2.circle(result_img, (cx, cy), radius, (0, 255, 255), 5)

                if show_numbers:
                    cv2.circle(result_img, (cx, cy), 22, (255, 255, 255), -1)
                    cv2.circle(result_img, (cx, cy), 22, (0, 0, 0), 2)
                    text = f"{i+1}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = cx - text_size[0] // 2
                    text_y = cy + text_size[1] // 2
                    cv2.putText(result_img, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if show_area_bool and has_geo:
                    area_m2 = area_pixels * detector.pixel_area_m2
                    area_text = f"{area_m2:.0f}m²"
                    text_size = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = cx - text_size[0] // 2
                    text_y = cy + radius + 20
                    cv2.rectangle(result_img,
                                (text_x - 3, text_y - text_size[1] - 3),
                                (text_x + text_size[0] + 3, text_y + 3),
                                (0, 0, 0), -1)
                    cv2.putText(result_img, area_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        h, w = results['before_img'].shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = results['before_img']
        comparison[:, w:] = result_img

        cv2.putText(comparison, "BEFORE", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(comparison, "AFTER (with detections)", (w + 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        num_buildings = len(results['new_buildings'])
        if has_geo:
            total_area = sum(cv2.contourArea(c) * detector.pixel_area_m2
                           for c in results['new_buildings'])
            stats = f"Found {num_buildings} new buildings | Total area: {total_area:.0f} m²"
        else:
            stats = f"Found {num_buildings} new buildings"

        return JSONResponse({
            "comparison_image": image_to_base64(comparison),
            "result_image": image_to_base64(result_img),
            "stats": stats,
            "num_buildings": num_buildings
        })

    except Exception as e:
        import traceback
        print(f"ERROR in change detection: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if before_path and os.path.exists(before_path):
            os.unlink(before_path)
        if after_path and os.path.exists(after_path):
            os.unlink(after_path)


@app.post("/api/parcel-analysis")
async def parcel_analysis(
    image: UploadFile = File(...),
    threshold: float = Form(0.05),
    min_area: int = Form(200),
    unused_threshold: int = Form(30)
):
    """
    Create synthetic land parcels around detected buildings
    """
    image_path = None

    try:
        image_path = save_upload_file(image)

        analyzer = SyntheticParcelAnalyzer()
        analyzer.load_image(image_path)
        analyzer.detect_buildings(threshold=threshold, min_area_pixels=min_area)

        if len(analyzer.buildings) == 0:
            raise HTTPException(status_code=400, detail="No buildings detected")

        analyzer.create_synthetic_parcels()

        result_img = analyzer.image.copy()

        for i, (building, parcel) in enumerate(zip(analyzer.buildings, analyzer.parcels)):
            x1, y1, x2, y2 = parcel['bbox_pixels']
            cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 255), 3)

            cx, cy = building['center']
            radius = int(np.sqrt(building['area_pixels'] / np.pi))
            cv2.circle(result_img, (cx, cy), radius, (255, 50, 50), -1)
            cv2.circle(result_img, (cx, cy), radius, (0, 255, 255), 2)

            color = (255, 0, 0) if parcel['flagged'] else (255, 255, 255)
            cv2.putText(result_img, f"{i+1}", (cx-10, cy+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        table_data = [{
            'Building #': i+1,
            'Building Area (m²)': f"{p['building_area_m2']:.1f}",
            'Parcel Area (m²)': f"{p['parcel_area_m2']:.1f}",
            'Unused (m²)': f"{p['unused_area_m2']:.1f}",
            'Unused %': f"{p['unused_percentage']:.1f}%",
            'Status': '⚠️ Flagged' if p['flagged'] else '✓ Normal'
        } for i, p in enumerate(analyzer.parcels)]

        total_building = sum(p['building_area_m2'] for p in analyzer.parcels)
        total_unused = sum(p['unused_area_m2'] for p in analyzer.parcels)
        flagged = sum(1 for p in analyzer.parcels if p['flagged'])

        stats = f"""📊 Analysis Results:
• Total Buildings: {len(analyzer.buildings)}
• Total Building Area: {total_building:.1f} m²
• Total Unused Area: {total_unused:.1f} m²
• Flagged Parcels: {flagged} ({flagged/len(analyzer.parcels)*100:.1f}%)"""

        # Convert parcels to JSON-serializable format (handle numpy types)
        parcels_json = []
        for p in analyzer.parcels:
            parcel_dict = {}
            for key, value in p.items():
                if isinstance(value, np.bool_):
                    parcel_dict[key] = bool(value)
                elif isinstance(value, (np.integer, np.floating)):
                    parcel_dict[key] = float(value)
                elif isinstance(value, np.ndarray):
                    parcel_dict[key] = value.tolist()
                else:
                    parcel_dict[key] = value
            parcels_json.append(parcel_dict)

        return JSONResponse({
            "visualization": image_to_base64(result_img),
            "stats": stats,
            "table_data": table_data,
            "chart_data": {
                "parcels": parcels_json
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"ERROR in parcel analysis: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
