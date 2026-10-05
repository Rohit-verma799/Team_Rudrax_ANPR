"""
Parallel OCR worker for processing license plate images.
Uses threading to run OCR without blocking the main detection loop.
"""

import os
# Disable PIR mode and MKL-DNN before any paddle imports
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import threading
import queue
import csv
import time
from datetime import datetime
from typing import Optional
import cv2
import numpy as np

from plate_ocr import PlateOCR


class OCRWorker:
    """
    Background worker that processes plate images in parallel.
    Uses a queue to receive images and processes them asynchronously.
    """
    
    def __init__(
        self,
        output_csv: str = "ocr_results.csv",
        use_gpu: bool = False,
        num_workers: int = 1
    ):
        """
        Initialize the OCR worker.
        
        Args:
            output_csv: Path to save OCR results
            use_gpu: Whether to use GPU for OCR
            num_workers: Number of worker threads (keep at 1 for PaddleOCR)
        """
        self.output_csv = output_csv
        self.use_gpu = use_gpu
        self.num_workers = num_workers
        
        # Queue for pending OCR tasks
        self.task_queue = queue.Queue()
        
        # Results storage
        self.results = []
        self.results_lock = threading.Lock()
        
        # Worker threads
        self.workers = []
        self.running = False
        
        # CSV file handle
        self.csv_file = None
        self.csv_writer = None
        
        # Statistics
        self.processed_count = 0
        self.start_time = None
    
    def start(self):
        """Start the OCR worker threads."""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Initialize CSV file
        self._init_csv()
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"OCRWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"OCR Worker started with {self.num_workers} thread(s)")
    
    def stop(self, wait: bool = True):
        """
        Stop the OCR worker.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if not self.running:
            return
        
        if wait:
            # Wait for queue to empty
            self.task_queue.join()
        
        self.running = False
        
        # Signal workers to stop
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"OCR Worker stopped. Processed {self.processed_count} plates in {elapsed:.1f}s")
    
    def submit(
        self,
        image: np.ndarray,
        vehicle_crop: np.ndarray,
        tracker_id: int,
        frame_num: int,
        vehicle_box: list,
        cropped_folder: str
    ):
        """
        Submit a plate image for OCR processing.
        
        Args:
            image: Plate image (BGR numpy array)
            vehicle_crop: Vehicle crop image (BGR numpy array)
            tracker_id: Vehicle tracker ID
            frame_num: Frame number
            vehicle_box: Vehicle bounding box [x1, y1, x2, y2]
            cropped_folder: Folder to save crops in
        """
        if not self.running:
            self.start()
        
        # Prepare file paths asynchronously
        v_filename = f"{tracker_id}_vehicle_f{frame_num}.jpg"
        vehicle_path = os.path.join(cropped_folder, v_filename)
        
        p_filename = f"{tracker_id}_plate_f{frame_num}.jpg"
        image_path = os.path.join(cropped_folder, p_filename)
        
        task = {
            'image': image.copy(),  # Copy to avoid reference issues
            'vehicle_crop': vehicle_crop.copy() if vehicle_crop is not None else None,
            'tracker_id': tracker_id,
            'frame_num': frame_num,
            'vehicle_box': vehicle_box,
            'image_path': image_path,
            'vehicle_path': vehicle_path,
            'timestamp': datetime.now().isoformat()
        }
        
        self.task_queue.put(task)
    
    def _init_csv(self):
        """Initialize the CSV output file."""
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(self.output_csv)
        
        self.csv_file = open(self.output_csv, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            # Write header row
            self.csv_writer.writerow([
                'timestamp',
                'frame_num',
                'tracker_id',
                'plate_text',
                'confidence',
                'vehicle_box',
                'image_path',
                'vehicle_path'
            ])
            self.csv_file.flush()
    
    def _worker_loop(self):
        """Main loop for worker thread."""
        # Initialize OCR engine in this thread with 2 CPU threads to prevent system starvation
        ocr = PlateOCR(use_gpu=self.use_gpu, cpu_threads=2)
        
        while self.running:
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:
                    # Shutdown signal
                    self.task_queue.task_done()
                    break
                
                # Process the task
                result = self._process_task(ocr, task)
                
                # Save result
                self._save_result(result)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"OCR Worker error: {e}")
                if task:
                    self.task_queue.task_done()
    
    def _process_task(self, ocr: PlateOCR, task: dict) -> dict:
        """
        Process a single OCR task.
        
        Args:
            ocr: PlateOCR instance
            task: Task dictionary with image and metadata
            
        Returns:
            Result dictionary
        """
        # Save crops to disk asynchronously in the background thread
        try:
            if 'vehicle_crop' in task and task['vehicle_crop'] is not None:
                cv2.imwrite(task['vehicle_path'], task['vehicle_crop'])
            if 'image' in task and task['image'] is not None:
                cv2.imwrite(task['image_path'], task['image'])
        except Exception as e:
            print(f"Error saving crop images asynchronously: {e}")
            
        image = task['image']
        
        # Run OCR
        ocr_result = ocr.extract_text(image)
        
        result = {
            'timestamp': task['timestamp'],
            'frame_num': task['frame_num'],
            'tracker_id': task['tracker_id'],
            'plate_text': ocr_result['text'],
            'confidence': ocr_result['confidence'],
            'vehicle_box': task['vehicle_box'],
            'image_path': task['image_path'],
            'vehicle_path': task['vehicle_path']
        }
        
        return result
    
    def _save_result(self, result: dict):
        """Save result to CSV and internal storage."""
        with self.results_lock:
            self.results.append(result)
            self.processed_count += 1
            
            # Write to CSV
            if self.csv_writer:
                self.csv_writer.writerow([
                    result['timestamp'],
                    result['frame_num'],
                    result['tracker_id'],
                    result['plate_text'],
                    result['confidence'],
                    str(result['vehicle_box']),
                    result['image_path'],
                    result['vehicle_path']
                ])
                self.csv_file.flush()
        
        # Print result
        plate_text = result['plate_text'] or "(no text)"
        print(f"[OCR] Tracker #{result['tracker_id']}: {plate_text} (conf: {result['confidence']:.2f})")
    
    def get_results(self) -> list:
        """Get all processed results."""
        with self.results_lock:
            return self.results.copy()
    
    def get_pending_count(self) -> int:
        """Get number of pending tasks in queue."""
        return self.task_queue.qsize()


# Singleton instance for easy access
_ocr_worker: Optional[OCRWorker] = None


def get_ocr_worker(
    output_csv: str = "ocr_results.csv",
    use_gpu: bool = False
) -> OCRWorker:
    """
    Get or create the global OCR worker instance.
    
    Args:
        output_csv: Path to save OCR results
        use_gpu: Whether to use GPU
        
    Returns:
        OCRWorker instance
    """
    global _ocr_worker
    
    if _ocr_worker is None:
        _ocr_worker = OCRWorker(output_csv=output_csv, use_gpu=use_gpu)
    
    return _ocr_worker


def shutdown_ocr_worker(wait: bool = True):
    """Shutdown the global OCR worker."""
    global _ocr_worker
    
    if _ocr_worker is not None:
        _ocr_worker.stop(wait=wait)
        _ocr_worker = None
