"""Video and webcam processing pipeline."""

import os
import cv2
import re
from collections import defaultdict

from vehicle_tracker import VehicleTracker
from plate_detector import PlateDetector
from ocr_worker import get_ocr_worker, shutdown_ocr_worker
from visualizer import Visualizer
from config import (
    DEFAULT_MODEL_PATH, DEFAULT_PLATE_MODEL_PATH,
    CROPPED_FOLDER, MAX_PLATE_DETECTIONS_PER_VEHICLE
)


class VideoProcessor:
    """Processes video/webcam with vehicle tracking and plate detection."""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence=0.8,
                 input_size=(384, 384), use_gpu=True, track_thresh=0.25,
                 track_buffer=30, match_thresh=0.8, plate_model=DEFAULT_PLATE_MODEL_PATH,
                 plate_confidence=0.8, enable_plates=True, cropped_folder=CROPPED_FOLDER,
                 max_plate_detections=MAX_PLATE_DETECTIONS_PER_VEHICLE,
                 zone_threshold=0.4):
        
        self.tracker = VehicleTracker(
            model_path, confidence, input_size, use_gpu,
            track_thresh, track_buffer, match_thresh
        )
        self.visualizer = Visualizer()
        self.enable_plates = enable_plates
        self.cropped_folder = cropped_folder
        self.max_plate_detections = max_plate_detections
        self.plate_counts = defaultdict(int)
        self.ocr_results = {}
        self.ocr_history = {}
        self.last_submitted_score = defaultdict(float)
        self.last_submitted_frame = defaultdict(int)
        self.zone_threshold = zone_threshold
        
        self.plate_detector = None
        self.plate_ocr = None
        if enable_plates:
            try:
                self.plate_detector = PlateDetector(
                    plate_model, plate_confidence, input_size, use_gpu
                )
                # Create cropped folder before initializing/starting workers
                os.makedirs(cropped_folder, exist_ok=True)
                
                # Initialize parallel OCR worker (always CPU to prevent GPU context/driver blocking stutters)
                self.ocr_worker = get_ocr_worker(
                    output_csv=os.path.join(cropped_folder, "ocr_results.csv"),
                    use_gpu=False
                )
                self.ocr_worker.start()
            except Exception as e:
                print(f"Plate detector unavailable: {e}")
                self.enable_plates = False

    def _detect_plates(self, frame, detections, frame_num):
        """Detect plates for vehicles in bottom half and keep the best."""
        if not self.enable_plates or detections.tracker_id is None:
            return []
        
        h = frame.shape[0]
        results = []
        
        for tid, box in zip(detections.tracker_id, detections.xyxy):
            x1, y1, x2, y2 = map(int, box)
            
            # Skip if not in plate detection zone
            if (y1 + y2) / 2 < h * self.zone_threshold:
                continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            plates = self.plate_detector.detect(crop)
            if plates:
                # Get the detection with the highest confidence score
                best_plate = max(plates, key=lambda x: x['score'])
                px1, py1, px2, py2 = best_plate['box']
                
                # Add to frame highlight results
                results.append({
                    'tracker_id': tid,
                    'vehicle_box': [x1, y1, x2, y2],
                    'plate_box': [x1 + px1, y1 + py1, x1 + px2, y1 + py2],
                    'plate_score': best_plate['score']
                })
                
                # We submit to OCR worker if:
                # 1. We haven't submitted anything yet for this tid.
                # 2. OR the new score is significantly better than the previously submitted score (at least 0.05 better).
                # 3. OR the score is slightly better (at least 0.01) and it's been at least 15 frames since the last submission.
                # Real-time submission condition: only run OCR in real-time if score >= 0.65
                # to filter out low-confidence/blurry/far-away plate detections.
                score = best_plate['score']
                last_score = self.last_submitted_score[tid]
                frames_since_last = frame_num - self.last_submitted_frame[tid]
                
                if score >= 0.65:
                    is_first = tid not in self.best_plates or last_score == 0.0
                    is_significant_improvement = (score >= last_score + 0.05)
                    is_periodic_upgrade = (score >= last_score + 0.01 and frames_since_last >= 15)
                    
                    if is_first or is_significant_improvement or is_periodic_upgrade:
                        px1, py1, px2, py2 = best_plate['box']
                        plate_crop = crop[py1:py2, px1:px2]
                        
                        if plate_crop.size > 0:
                            # Update tracking variables
                            self.last_submitted_score[tid] = score
                            self.last_submitted_frame[tid] = frame_num
                            
                            # Submit to background OCR worker with raw crops (file saving is done asynchronously in the background thread)
                            self.ocr_worker.submit(
                                image=plate_crop,
                                vehicle_crop=crop,
                                tracker_id=tid,
                                frame_num=frame_num,
                                vehicle_box=[x1, y1, x2, y2],
                                cropped_folder=self.cropped_folder
                            )
                
                # Always save/update the best crop overall in self.best_plates for tracking and fallback
                if tid not in self.best_plates or score > self.best_plates[tid]['score']:
                    px1, py1, px2, py2 = best_plate['box']
                    plate_crop = crop[py1:py2, px1:px2]
                    
                    if plate_crop.size > 0:
                        self.best_plates[tid] = {
                            'score': score,
                            'frame_num': frame_num,
                            'vehicle_box': [x1, y1, x2, y2],
                            'plate_crop': plate_crop.copy(),
                            'vehicle_crop': crop.copy()  # Crop is the cropped vehicle image
                        }
                        self.plate_counts[tid] = 1  # Unique plate count for statistics
        
        return results

    def process_video(self, video_path, output_path=None, show=False, show_trace=True):
        """Process video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.tracker.reset()
        self.tracker.set_frame_rate(fps)
        self.plate_counts.clear()
        self.ocr_results.clear()
        self.ocr_history.clear()
        self.best_plates = {}  # Initialize best plates cache
        
        writer = None
        if output_path:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
        if show:
            # Scale video size to fit in a 1024x720 window while maintaining aspect ratio
            max_w, max_h = 1024, 720
            scale = min(max_w / w, max_h / h)
            display_w = int(w * scale)
            display_h = int(h * scale)
            cv2.namedWindow("Vehicle Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Vehicle Tracking", display_w, display_h)
        
        frame_num = 0
        all_tracks = set()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                detections = self.tracker.track(frame)
                
                if detections.tracker_id is not None:
                    all_tracks.update(detections.tracker_id.tolist())
                
                plate_results = self._detect_plates(frame, detections, frame_num)
                
                # Poll background OCR results
                if self.enable_plates:
                    self.ocr_history.clear()
                    for res in self.ocr_worker.get_results():
                        tid = res['tracker_id']
                        if not res['plate_text']:
                            continue
                        if tid not in self.ocr_history:
                            self.ocr_history[tid] = []
                        self.ocr_history[tid].append({
                            'text': res['plate_text'],
                            'confidence': res['confidence']
                        })
                    
                    # Re-evaluate best candidate for each vehicle using multi-frame voting consensus
                    for tid, history in self.ocr_history.items():
                        candidates = {}
                        for item in history:
                            candidate_text = item['text']
                            if candidate_text not in candidates:
                                candidates[candidate_text] = {
                                    'count': 0,
                                    'conf_sum': 0.0
                                }
                            candidates[candidate_text]['count'] += 1
                            candidates[candidate_text]['conf_sum'] += item['confidence']
                            
                        best_candidate = None
                        best_score = -1.0
                        
                        for text, stats in candidates.items():
                            freq = stats['count']
                            avg_conf = stats['conf_sum'] / freq
                            score = freq * avg_conf
                            
                            # Standard layout format boosts:
                            # 1. Standard State District Series Number format (e.g., MP 04 ZX 1632)
                            has_standard_format = bool(re.match(r'^[A-Z]{2}\s+\d{1,2}\s+[A-Z]{1,3}\s+\d{4}$', text))
                            # 2. BH series format: YY BH #### XX (e.g. 24 BH 6432 O)
                            has_bh_format = bool(re.match(r'^\d{2}\s+BH\s+\d{4}\s+[A-Z]{1,2}$', text))
                            
                            if has_standard_format or has_bh_format:
                                score += 0.2
                                
                            if score > best_score:
                                best_score = score
                                best_candidate = {
                                    'text': text,
                                    'confidence': avg_conf
                                }
                                
                        if best_candidate:
                            old_val = self.ocr_results.get(tid)
                            self.ocr_results[tid] = best_candidate
                            
                            # Print consensus update to terminal when the active display plate text changes
                            if old_val is None or old_val['text'] != best_candidate['text']:
                                print(f"[Consensus Update] Tracker #{tid}: {best_candidate['text']} (conf: {best_candidate['confidence']:.2f})")
                
                # Annotate
                annotated = self.visualizer.annotate(frame, detections, show_trace)
                self.visualizer.draw_zone_line(annotated, int(h * self.zone_threshold), "Plate Detection Zone")
                
                for pr in plate_results:
                    self.visualizer.highlight_box(annotated, pr['vehicle_box'])
                    if 'plate_box' in pr:
                        # Draw a cyan box around the plate itself
                        self.visualizer.highlight_box(annotated, pr['plate_box'], color=(255, 255, 0), thickness=2)
                        px1, py1, px2, py2 = pr['plate_box']
                        cv2.putText(
                            annotated,
                            f"Plate: {pr['plate_score']:.2f}",
                            (px1, max(15, py1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            2
                        )
                        
                        # Real-time OCR text overlay
                        tid = pr['tracker_id']
                        if tid in self.ocr_results:
                            ocr_text = self.ocr_results[tid]['text']
                            cv2.putText(
                                annotated,
                                ocr_text,
                                (px1, py2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2
                            )
                
                plates_saved = sum(self.plate_counts.values())
                self.visualizer.draw_stats(
                    annotated,
                    f"Frame: {frame_num}/{total} | Vehicles: {len(detections)} | Plates: {plates_saved}"
                )
                
                if writer:
                    writer.write(annotated)
                if show:
                    cv2.imshow("Vehicle Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if frame_num % 30 == 0:
                    print(f"Frame {frame_num}/{total} | Vehicles: {len(detections)} | Plates: {plates_saved}")
            
            # Fallback: Submit any vehicle crops that were never submitted to OCR in real-time (e.g. score was always < 0.65)
            if self.enable_plates and self.best_plates:
                for tid, data in self.best_plates.items():
                    if self.last_submitted_score[tid] == 0.0:
                        self.last_submitted_score[tid] = data['score']
                        self.ocr_worker.submit(
                            image=data['plate_crop'],
                            vehicle_crop=data['vehicle_crop'],
                            tracker_id=tid,
                            frame_num=data['frame_num'],
                            vehicle_box=data['vehicle_box'],
                            cropped_folder=self.cropped_folder
                        )
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            # Shutdown OCR worker and wait for pending tasks
            if self.enable_plates and hasattr(self, 'ocr_worker'):
                print("Waiting for OCR to complete...")
                shutdown_ocr_worker(wait=True)
        
        self._print_summary(frame_num, all_tracks, output_path)

    def process_webcam(self, camera_id=0, show_trace=True):
        """Process live webcam feed."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        self.tracker.reset()
        all_tracks = set()
        
        print("Press 'q' to quit, 'r' to reset")
        cv2.namedWindow("Vehicle Tracking", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections = self.tracker.track(frame)
                
                if detections.tracker_id is not None:
                    all_tracks.update(detections.tracker_id.tolist())
                
                annotated = self.visualizer.annotate(frame, detections, show_trace)
                self.visualizer.draw_stats(
                    annotated, f"Active: {len(detections)} | Total: {len(all_tracks)}"
                )
                
                cv2.imshow("Vehicle Tracking", annotated)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.tracker.reset()
                    all_tracks.clear()
                    print("Reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            # Shutdown OCR worker and wait for pending tasks
            if self.enable_plates and hasattr(self, 'ocr_worker'):
                print("Waiting for OCR to complete...")
                shutdown_ocr_worker(wait=True)
        
        print(f"Total vehicles tracked: {len(all_tracks)}")

    def _print_summary(self, frames, tracks, output_path):
        """Print processing summary."""
        print(f"\n{'='*40}")
        print(f"Frames processed: {frames}")
        print(f"Unique vehicles: {len(tracks)}")
        print(f"Plates saved: {sum(self.plate_counts.values())}")
        print(f"Vehicles with plates: {len(self.plate_counts)}")
        if output_path:
            print(f"Output: {output_path}")
        if self.enable_plates:
            print(f"Plates folder: {self.cropped_folder}")
            print(f"OCR results: {os.path.join(self.cropped_folder, 'ocr_results.csv')}")
        print('='*40)

    def process_image(self, image_path, output_path=None, show=False):
        """Process a single image for vehicle and plate detection + OCR."""
        self.process_image_single(image_path, output_path, show)
        # Clean shutdown of worker if running in single-shot mode
        if self.enable_plates and hasattr(self, 'ocr_worker'):
            shutdown_ocr_worker(wait=True)

    def process_image_folder(self, folder_path, output_folder=None, show=False):
        """Process a folder of images for vehicle and plate detection + OCR."""
        import glob
        
        # Supported image formats
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
        image_files = list(set(image_files))  # Deduplicate
        image_files.sort()
        
        if not image_files:
            print(f"No images found in folder: {folder_path}")
            return
            
        print(f"Found {len(image_files)} images in folder to process.")
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            
        for idx, img_path in enumerate(image_files, 1):
            img_name = os.path.basename(img_path)
            out_path = os.path.join(output_folder, img_name) if output_folder else None
            
            print(f"\n[{idx}/{len(image_files)}] Processing: {img_name}")
            self.process_image_single(img_path, out_path, show=show, frame_id=idx)
            
        # Shut down worker at the very end of the batch
        if self.enable_plates and hasattr(self, 'ocr_worker'):
            shutdown_ocr_worker(wait=True)

    def process_image_single(self, image_path, output_path=None, show=False, frame_id=1):
        """Process a single image for vehicle and plate detection + OCR without shutting down worker."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from '{image_path}'")
            return
        
        self.tracker.reset()
        self.plate_counts.clear()
        self.ocr_results.clear()
        self.ocr_history.clear()
        self.best_plates = {}
        
        # Detect vehicles
        detections = self.tracker.detect(frame)
        if len(detections) > 0:
            import numpy as np
            # Assign dummy tracker IDs so downstream OCR worker and results logic functions correctly
            detections.tracker_id = np.arange(1, len(detections) + 1)
        
        # Save original zone threshold and temporarily set to 0.0 to detect plates on all vehicles in the image
        orig_zone_thresh = self.zone_threshold
        self.zone_threshold = 0.0
        
        try:
            plate_results = self._detect_plates(frame, detections, frame_num=frame_id)
            
            # Wait for background OCR worker threads to complete processing submitted tasks
            if self.enable_plates and hasattr(self, 'ocr_worker'):
                self.ocr_worker.task_queue.join()
                
                # Fetch results from the worker and populate self.ocr_results
                for res in self.ocr_worker.get_results():
                    if res['frame_num'] == frame_id:
                        tid = res['tracker_id']
                        if not res['plate_text']:
                            continue
                        self.ocr_results[tid] = {
                            'text': res['plate_text'],
                            'confidence': res['confidence']
                        }
        finally:
            self.zone_threshold = orig_zone_thresh
            
        # Annotate
        annotated = self.visualizer.annotate(frame, detections, show_trace=False)
        
        for pr in plate_results:
            self.visualizer.highlight_box(annotated, pr['vehicle_box'])
            if 'plate_box' in pr:
                # Draw a yellow box around the plate itself
                self.visualizer.highlight_box(annotated, pr['plate_box'], color=(255, 255, 0), thickness=2)
                px1, py1, px2, py2 = pr['plate_box']
                cv2.putText(
                    annotated,
                    f"Plate: {pr['plate_score']:.2f}",
                    (px1, max(15, py1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )
                
                # Real-time OCR text overlay
                tid = pr['tracker_id']
                if tid in self.ocr_results:
                    ocr_text = self.ocr_results[tid]['text']
                    cv2.putText(
                        annotated,
                        ocr_text,
                        (px1, py2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
                    
        if output_path:
            # Ensure the output directory exists
            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(output_path, annotated)
            print(f"Processed image saved to: {output_path}")
            
        if show:
            # Scale visualizer window if image is too large
            h, w = frame.shape[:2]
            max_w, max_h = 1024, 720
            scale = min(max_w / w, max_h / h)
            display_w = int(w * scale)
            display_h = int(h * scale)
            cv2.namedWindow("Image OCR Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image OCR Result", display_w, display_h)
            cv2.imshow("Image OCR Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        # Print a concise summary
        self._print_summary(frames=1, tracks=set(detections.tracker_id.tolist()) if len(detections) > 0 else set(), output_path=output_path)
