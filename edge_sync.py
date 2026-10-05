"""
edge_sync.py - Edge Transmit Client for NVIDIA Jetson Orin Nano
Sends license plate recognition events from edge gate devices to central API backend.
"""

import os
import requests
import json
import base64
import time
from datetime import datetime

class EdgeSyncClient:
    def __init__(self, api_url="http://localhost:8000/api/v1/vehicles/event", gate_id="GATE_ENTRY_01", gate_type="ENTRY"):
        self.api_url = api_url
        self.gate_id = gate_id
        self.gate_type = gate_type  # 'ENTRY' or 'EXIT'
        
    def send_ocr_event(self, tracker_id, plate_text_en, plate_text_ar, confidence, image_path, vehicle_box=None):
        """
        Encodes plate crop image and transmits OCR event payload to Central Backend.
        """
        try:
            image_base64 = ""
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            payload = {
                "gate_id": self.gate_id,
                "gate_type": self.gate_type,
                "tracker_id": str(tracker_id),
                "plate_number_en": plate_text_en or "",
                "plate_number_ar": plate_text_ar or "",
                "confidence": float(confidence or 0.0),
                "timestamp": datetime.now().isoformat(),
                "vehicle_box": vehicle_box or [0, 0, 0, 0],
                "image_base64": image_base64
            }
            
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.api_url, data=json.dumps(payload), headers=headers, timeout=3.0)
            
            if response.status_code == 200:
                print(f"[EDGE SYNC SUCCESS] Event for Plate '{plate_text_en}' ({self.gate_type}) synced to backend.")
                return True
            else:
                print(f"[EDGE SYNC ERROR] HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"[EDGE SYNC FAILED] Could not transmit from Jetson Edge Node: {e}")
            return False

if __name__ == "__main__":
    # Test execution
    client = EdgeSyncClient(gate_id="GATE_01_TEST", gate_type="ENTRY")
    client.send_ocr_event(
        tracker_id=101,
        plate_text_en="ABC-1234",
        plate_text_ar="أ ب ج 1234",
        confidence=0.95,
        image_path="",
        vehicle_box=[100, 200, 300, 400]
    )
