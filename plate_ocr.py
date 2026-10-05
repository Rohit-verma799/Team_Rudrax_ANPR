"""
PaddleOCR-based license plate text recognition.
Supports both Arabic and English text extraction.
"""

import os
import re

# Determine if we should use PIR mode (needed for Paddle 3.x / PaddleOCR 3.x)
try:
    import paddle
    # Check if we have paddleocr 3.x (which imports TextRecognition)
    from paddleocr import TextRecognition
    HAS_TEXT_RECOGNITION = True
except ImportError:
    HAS_TEXT_RECOGNITION = False

if HAS_TEXT_RECOGNITION:
    os.environ['FLAGS_enable_pir_api'] = '1'
    os.environ['FLAGS_enable_pir_in_executor'] = '1'
else:
    # Disable PIR mode for older Paddle versions to fix compatibility issues
    os.environ['FLAGS_enable_pir_api'] = '0'
    os.environ['FLAGS_enable_pir_in_executor'] = '0'

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import paddle
paddle.set_flags({'FLAGS_use_mkldnn': 0})

from paddleocr import PaddleOCR
try:
    from paddleocr import TextRecognition
except ImportError:
    TextRecognition = None

import numpy as np
from typing import Optional, List, Tuple
import cv2


class PlateOCR:
    """OCR engine for license plate text recognition using PaddleOCR."""
    
    def __init__(self, use_gpu: bool = True, cpu_threads: int = 2):
        """
        Initialize PaddleOCR engine. If custom OCRNEW model directory is present,
        use it with its corresponding dictionary.
        
        Args:
            use_gpu: Whether to use GPU acceleration (default: False for CPU)
            cpu_threads: Number of CPU threads to use (default: 2)
        """
        custom_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr2")
        inference_yml = os.path.join(custom_model_dir, "inference.yml")
        
        if os.path.exists(custom_model_dir) and os.path.exists(inference_yml):
            print(f"Custom OCR model found in {custom_model_dir}. Preparing custom dictionary...")
            try:
                import yaml
                with open(inference_yml, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                char_dict = config.get('PostProcess', {}).get('character_dict', [])
                
                if char_dict:
                    dict_path = os.path.join(custom_model_dir, "dict.txt")
                    with open(dict_path, 'w', encoding='utf-8') as f:
                        for char in char_dict:
                            f.write(char + '\n')
                    print(f"Generated custom dictionary at {dict_path}")
                    
                    if HAS_TEXT_RECOGNITION:
                        print("Using PaddleOCR 3.x TextRecognition engine for custom model.")
                        self.ocr_en = TextRecognition(
                            model_dir=custom_model_dir,
                            device="gpu" if use_gpu else "cpu",
                            enable_mkldnn=False
                        )
                    else:
                        print("Using PaddleOCR 2.x engine for custom model.")
                        self.ocr_en = PaddleOCR(
                            use_textline_orientation=True,
                            lang='en',
                            use_gpu=use_gpu,
                            rec_model_dir=custom_model_dir,
                            rec_char_dict_path=dict_path,
                            cpu_threads=cpu_threads
                        )
                    return
            except Exception as e:
                print(f"Error loading custom OCR model: {e}. Falling back to default OCR.")
        
        # Fallback to default English OCR engine
        if HAS_TEXT_RECOGNITION:
            print("Using PaddleOCR 3.x engine for default English OCR.")
            self.ocr_en = PaddleOCR(
                use_textline_orientation=True,
                device="gpu" if use_gpu else "cpu"
            )
        else:
            print("Using PaddleOCR 2.x engine for default English OCR.")
            self.ocr_en = PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                use_gpu=use_gpu,
                cpu_threads=cpu_threads
            )

    
    def preprocess_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR results.
        Resizes the plate image maintaining aspect ratio for optimal PaddleOCR height.
        
        Args:
            image: BGR plate image
            
        Returns:
            Preprocessed image
        """
        if image is None or image.size == 0:
            return image

        # Standardize height to 120 pixels for better PaddleOCR character extraction
        h, w = image.shape[:2]
        if h < 120:
            scale = 120.0 / h
            new_w = int(w * scale)
            resized = cv2.resize(image, (new_w, 120), interpolation=cv2.INTER_CUBIC)
            return resized
        else:
            return image.copy()
    
    def extract_text(
        self, 
        image: np.ndarray, 
        preprocess: bool = True
    ) -> dict:
        """
        Extract text from license plate image using English OCR.
        
        Args:
            image: BGR plate image (numpy array)
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary containing:
                - 'text': Recognized text
                - 'english': English text results
                - 'arabic': Empty string (Arabic disabled)
                - 'confidence': Average confidence score
                - 'details': Raw OCR details
        """
        if image is None or image.size == 0:
            return {
                'text': '',
                'english': '',
                'arabic': '',
                'confidence': 0.0,
                'details': []
            }
        
        # Preprocess if requested
        img_to_process = self.preprocess_plate(image) if preprocess else image
        
        # Run English OCR
        en_results = self._run_ocr(self.ocr_en, img_to_process)
        
        # Combined text
        text_results = ' '.join([r[0] for r in en_results])
        cleaned_text = self.clean_indian_plate(text_results)
        
        # Calculate average confidence
        all_confidences = [r[1] for r in en_results]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        return {
            'text': cleaned_text,
            'raw_text': text_results,
            'english': cleaned_text,
            'arabic': '',
            'confidence': round(avg_confidence, 3),
            'details': {
                'english_results': en_results,
                'arabic_results': []
            }
        }
    
    def _run_ocr(
        self, 
        ocr_engine: PaddleOCR, 
        image: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Run OCR engine on image.
        Runs both detection-enabled and recognition-only modes,
        and selects the one with the higher average confidence.
        
        Args:
            ocr_engine: PaddleOCR instance
            image: Preprocessed image
            
        Returns:
            List of (text, confidence) tuples
        """
        if TextRecognition is not None and isinstance(ocr_engine, TextRecognition):
            try:
                res = ocr_engine.predict(image)
                if res and len(res) > 0:
                    text = res[0].get('rec_text', '').strip()
                    score = res[0].get('rec_score', 0.0)
                    if text:
                        return [(text, score)]
            except Exception as e:
                print(f"TextRecognition predict error: {e}")
            return []

        results_rec = []
        avg_conf_rec = 0.0
        try:
            # 1. Try with recognition-only (det=False) first as it is much faster
            rec_output = ocr_engine.ocr(image, det=False)
            if rec_output and rec_output[0]:
                for line in rec_output[0]:
                    if isinstance(line, tuple) and len(line) == 2:
                        text, confidence = line
                        if text.strip():
                            results_rec.append((text.strip(), confidence))
                if results_rec:
                    avg_conf_rec = sum(r[1] for r in results_rec) / len(results_rec)
        except Exception as e:
            print(f"OCR det=False error: {e}")

        # Early return if det=False yielded a highly confident result to save time
        if avg_conf_rec >= 0.90:
            return results_rec

        results_det = []
        avg_conf_det = 0.0
        try:
            # 2. Try with detection enabled (det=True) as fallback
            ocr_output = ocr_engine.ocr(image, det=True)
            if ocr_output and ocr_output[0]:
                for line in ocr_output[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        if text.strip():
                            results_det.append((text.strip(), confidence))
                if results_det:
                    avg_conf_det = sum(r[1] for r in results_det) / len(results_det)
        except Exception as e:
            print(f"OCR det=True error: {e}")

        # Choose the mode with higher average confidence
        if avg_conf_rec > avg_conf_det:
            return results_rec
        else:
            return results_det
    
    def _combine_results(
        self, 
        en_results: List[Tuple[str, float]], 
        ar_results: List[Tuple[str, float]]
    ) -> str:
        """
        Combine English and Arabic results intelligently.
        
        For license plates, typically the format includes both numbers/letters
        and potentially Arabic text. We prioritize higher confidence results.
        
        Args:
            en_results: English OCR results
            ar_results: Arabic OCR results
            
        Returns:
            Combined text string
        """
        # If one is empty, return the other
        if not en_results and not ar_results:
            return ''
        if not en_results:
            return ' '.join([r[0] for r in ar_results])
        if not ar_results:
            return ' '.join([r[0] for r in en_results])
        
        # Use the result with higher average confidence
        en_avg = sum([r[1] for r in en_results]) / len(en_results)
        ar_avg = sum([r[1] for r in ar_results]) / len(ar_results)
        
        # If confidences are close, combine both
        if abs(en_avg - ar_avg) < 0.1:
            en_text = ' '.join([r[0] for r in en_results])
            ar_text = ' '.join([r[0] for r in ar_results])
            return f"{en_text} | {ar_text}"
        
        # Return higher confidence result
        if en_avg > ar_avg:
            return ' '.join([r[0] for r in en_results])
        else:
            return ' '.join([r[0] for r in ar_results])
            
    def clean_indian_plate(self, text: str) -> str:
        """
        Cleans and formats Indian license plate numbers by removing noise
        (e.g., HSRP 'IND' badge, screw marks, borders) and correcting OCR errors.
        Supports both single-line and double-line layout formats.
        """
        if not text:
            return ""
            
        text = text.upper().strip()
        words = re.findall(r'[A-Z0-9]+', text)
        if not words:
            return text
            
        cleaned_words = []
        for w in words:
            # Strip HSRP 'IND' noise prefix from the word
            w_clean = re.sub(r'^(1?ND|ON[D0]|IID|IND|IN|1N)[0-9O]?', '', w)
            if w_clean:
                cleaned_words.append(w_clean)
                
        if not cleaned_words:
            cleaned_words = words
            
        # Check if this is a BH (Bharat) Series plate
        # Format: YY BH #### XX (e.g., 24 BH 6432 D)
        bh_flat = "".join(cleaned_words)
        bh_match = re.search(r'^(\d{2})(?:BH|8H|B1|B|H)([0-9OQDI L]{4})([A-Z0-9]{1,2})$', bh_flat)
        if bh_match:
            yy = bh_match.group(1)
            
            # Helper to correct common OCR substitutions to digits
            def to_digits_local(s):
                s = s.replace('O', '0').replace('Q', '0').replace('D', '0')
                s = s.replace('I', '1').replace('L', '1').replace('T', '1').replace('J', '1')
                s = s.replace('Z', '2').replace('S', '5').replace('B', '8')
                return "".join(c for c in s if c.isdigit())
                
            num_part = to_digits_local(bh_match.group(2))
            if len(num_part) > 4:
                num_part = num_part[:4]
                
            # Suffix is 1 or 2 letters. Map any incorrectly recognized digits back to letters.
            suffix_part = bh_match.group(3)
            suffix_letters = []
            for c in suffix_part:
                if c == '0':
                    suffix_letters.append('O')
                elif c == '1':
                    suffix_letters.append('I')
                elif c == '2':
                    suffix_letters.append('Z')
                elif c == '5':
                    suffix_letters.append('S')
                elif c == '8':
                    suffix_letters.append('B')
                else:
                    suffix_letters.append(c)
            suffix_clean = "".join(suffix_letters)
            return f"{yy} BH {num_part} {suffix_clean}"
            
        valid_state_codes = {
            "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
            "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
            "TR", "UP", "UK", "WB", "AN", "CH", "DN", "DD", "DL", "JK", "LA", "LD",
            "PY", "UA"
        }
        
        # Fuzzy and exact state code matching helper
        def get_fuzzy_state(word: str) -> Optional[Tuple[str, int]]:
            if len(word) < 2:
                return None
                
            char_maps = {
                '1': ['L', 'I', 'T', 'J'],
                '0': ['O', 'D', 'Q'],
                '8': ['B', 'R'],
                '5': ['S'],
                '2': ['Z'],
                '3': ['E'],
                '4': ['A'],
                '6': ['G'],
                '7': ['T'],
                '9': ['P']
            }
            
            # Check starting prefix
            prefix = word[:2]
            c0_opts = char_maps.get(prefix[0], [prefix[0]])
            c1_opts = char_maps.get(prefix[1], [prefix[1]])
            for c0 in c0_opts:
                for c1 in c1_opts:
                    candidate = c0 + c1
                    if candidate in valid_state_codes:
                        return candidate, 0
                        
            # Check sub-segments
            for idx in range(len(word) - 1):
                c0_opts = char_maps.get(word[idx], [word[idx]])
                c1_opts = char_maps.get(word[idx+1], [word[idx+1]])
                for c0 in c0_opts:
                    for c1 in c1_opts:
                        candidate = c0 + c1
                        if candidate in valid_state_codes:
                            return candidate, idx
            return None

        # Find and correct the state code
        state = ""
        state_word_idx = -1
        state_char_idx = -1
        
        for i, w in enumerate(cleaned_words):
            match = get_fuzzy_state(w)
            if match:
                state, idx = match
                state_word_idx = i
                state_char_idx = idx
                # Update word with corrected state code prefix/segment
                cleaned_words[i] = w[:idx] + state + w[idx+2:]
                break
                
        if not state:
            # Fallback to alphanumeric cleaning if no state code found
            cleaned = "".join(c for c in "".join(cleaned_words) if c.isalnum())
            return cleaned
            
        # Helper to correct common OCR substitutions
        def to_digits(s):
            s = s.replace('O', '0').replace('Q', '0').replace('D', '0')
            s = s.replace('I', '1').replace('L', '1').replace('T', '1').replace('J', '1')
            s = s.replace('Z', '2').replace('S', '5').replace('B', '8')
            return "".join(c for c in s if c.isdigit())
            
        district = ""
        series = ""
        number = ""
        
        suffix = cleaned_words[state_word_idx][state_char_idx + 2:]
        remaining_words = cleaned_words[state_word_idx + 1:]
        
        if suffix:
            # Suffix could contain District, Series, Number
            dist_match = re.match(r'^([0-9OQDI L]{1,2})', suffix)
            if dist_match:
                district = to_digits(dist_match.group(1))
                suffix = suffix[len(dist_match.group(1)):]
            else:
                dist_match = re.search(r'([0-9OQDI L]{1,2})', suffix)
                if dist_match:
                    district = to_digits(dist_match.group(1))
                    suffix = suffix[dist_match.end():]
                    
            if suffix:
                match = re.match(r'^([A-Z]{1,3})?([0-9OQDI L]{1,4})?', suffix)
                if match:
                    if match.group(1):
                        series = match.group(1)
                    if match.group(2):
                        number = to_digits(match.group(2))
        
        # 2. Extract District from subsequent words if not found in state word
        word_ptr = 0
        if not district and remaining_words:
            word = remaining_words[word_ptr]
            digits = to_digits(word)
            if digits:
                district = digits[:2]
                word_ptr += 1
                
        # 3. Process remaining words to fill series and number
        while word_ptr < len(remaining_words):
            word = remaining_words[word_ptr]
            word_ptr += 1
            
            match = re.match(r'^([A-Z]+)?([0-9OQDI L]+)?', word)
            if match:
                w_letters = match.group(1) or ""
                w_digits = to_digits(match.group(2) or "")
                
                if w_letters and w_digits:
                    # e.g., "N7704" or "AB1234"
                    if not series:
                        series = w_letters
                    number = w_digits
                elif w_letters:
                    # e.g., "AB"
                    if not series:
                        series = w_letters
                    else:
                        series += w_letters
                elif w_digits:
                    # e.g., "1234" or "55464"
                    if not number:
                        number = w_digits
                    else:
                        if not district:
                            district = w_digits[:2]
                        else:
                            number += w_digits
                            
        # Standardize district to at most 2 digits
        if len(district) > 2:
            district = district[:2]
            
        # Standardize number to at most 4 digits (taking first 4)
        if len(number) > 4:
            number = number[:4]
            
        if district and number:
            parts = [state, district]
            if series:
                parts.append(series[:3])
            parts.append(number)
            return " ".join(parts)
            
        # Fallback to general cleaning
        combined = "".join(cleaned_words[state_word_idx:])
        cleaned = "".join(c for c in combined if c.isalnum())
        return cleaned
    
    def extract_text_from_file(self, image_path: str) -> dict:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to plate image file
            
        Returns:
            OCR results dictionary
        """
        image = cv2.imread(image_path)
        if image is None:
            return {
                'text': '',
                'english': '',
                'arabic': '',
                'confidence': 0.0,
                'details': [],
                'error': f'Failed to load image: {image_path}'
            }
        
        return self.extract_text(image)


if __name__ == "__main__":
    import glob
    import re
    import csv
    from datetime import datetime
    from config import CROPPED_FOLDER
    
    print(f"Scanning folder: {CROPPED_FOLDER}")
    if not os.path.exists(CROPPED_FOLDER):
        os.makedirs(CROPPED_FOLDER, exist_ok=True)
        print("Folder created. No images to process.")
        exit(0)
        
    image_paths = glob.glob(os.path.join(CROPPED_FOLDER, "*.jpg"))
    if not image_paths:
        print("No cropped plate images found in folder.")
        exit(0)
        
    output_csv = os.path.join(CROPPED_FOLDER, "ocr_results.csv")
    file_exists = os.path.exists(output_csv)
    
    # Initialize PlateOCR
    print("Initializing OCR engine...")
    ocr = PlateOCR(use_gpu=False)  # default to CPU
    
    # Read existing entries to avoid processing duplicates
    processed_files = set()
    if file_exists:
        try:
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 7:
                        # row[6] is image_path
                        processed_files.add(os.path.basename(row[6]))
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            
    # Open CSV for appending
    with open(output_csv, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        if not file_exists:
            writer.writerow([
                'timestamp',
                'frame_num',
                'tracker_id',
                'plate_text',
                'confidence',
                'vehicle_box',
                'image_path',
                'vehicle_path'
            ])
            csv_file.flush()
            
        processed_count = 0
        for path in image_paths:
            filename = os.path.basename(path)
            
            # Skip vehicle crops and already processed files
            if "_vehicle_" in filename or filename in processed_files:
                continue
                
            print(f"Processing plate crop: {filename}")
            
            # Parse tracker_id and frame_num from filename
            # filename format: {tracker_id}_plate_f{frame_num}.jpg
            # fallback format: {tracker_id}_{count}_f{frame_num}.jpg
            tracker_id = -1
            frame_num = -1
            vehicle_path = ""
            
            match_new = re.search(r"(\d+)_plate_f(\d+)\.jpg", filename)
            if match_new:
                tracker_id = int(match_new.group(1))
                frame_num = int(match_new.group(2))
                # Resolve vehicle crop path
                vehicle_filename = f"{tracker_id}_vehicle_f{frame_num}.jpg"
                v_path = os.path.join(CROPPED_FOLDER, vehicle_filename)
                if os.path.exists(v_path):
                    vehicle_path = v_path
            else:
                match = re.search(r"(\d+)_(\d+)_f(\d+)\.jpg", filename)
                if match:
                    tracker_id = int(match.group(1))
                    frame_num = int(match.group(3))
                else:
                    match_fallback = re.search(r"(\d+)_(\d+)_frame(\d+)\.jpg", filename)
                    if match_fallback:
                        tracker_id = int(match_fallback.group(1))
                        frame_num = int(match_fallback.group(3))
            
            # Extract text
            res = ocr.extract_text_from_file(path)
            
            # Write to CSV
            writer.writerow([
                datetime.now().isoformat(),
                frame_num,
                tracker_id,
                res.get('text', ''),
                res.get('confidence', 0.0),
                str([]),  # vehicle_box is empty for offline mode
                path,
                vehicle_path
            ])
            csv_file.flush()
            processed_count += 1
            print(f"OCR result: {res.get('text', '')} (conf: {res.get('confidence', 0.0)})")
            
    print(f"Completed! Processed {processed_count} new images.")


