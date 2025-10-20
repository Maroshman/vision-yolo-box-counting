"""
Label Processing Module

Handles barcode/QR code detection and OCR on detected labels.
Uses OpenCV's built-in detectors by default.
Optional: Can use pyzbar for better barcode detection (set USE_PYZBAR env var).
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
import logging

# Import after other imports to ensure .env is loaded by the calling module
def _check_pyzbar():
    """Check if pyzbar should be used based on environment variable."""
    import os
    from dotenv import load_dotenv
    load_dotenv()  # Ensure .env is loaded
    
    use_pyzbar = os.getenv("USE_PYZBAR", "false").lower() in ("true", "1", "yes")
    
    if use_pyzbar:
        try:
            from pyzbar import pyzbar
            return True, pyzbar
        except ImportError:
            return False, None
    return False, None

# Initialize pyzbar if requested
PYZBAR_AVAILABLE, pyzbar = _check_pyzbar()


class LabelProcessor:
    """
    Process label images to extract barcodes, QR codes, and text.
    """
    
    def __init__(self, use_ocr: bool = True, ocr_languages: List[str] = ['en'], use_gpu: bool = False):
        """
        Initialize label processor.
        
        Args:
            use_ocr: Whether to use OCR as fallback
            ocr_languages: Languages for OCR (default: ['en'])
            use_gpu: Whether to use GPU for OCR
        """
        self.use_ocr = use_ocr
        self.ocr_languages = ocr_languages
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(__name__)
        self.use_pyzbar = PYZBAR_AVAILABLE
        
        if self.use_pyzbar:
            self.logger.info("Using pyzbar for barcode detection (more reliable)")
        else:
            self.logger.info("Using OpenCV for barcode detection (set USE_PYZBAR=true for better detection)")
        
        # Initialize OpenCV barcode detector (used if pyzbar not enabled)
        if not self.use_pyzbar:
            try:
                self.barcode_detector = cv2.barcode.BarcodeDetector()
                self.logger.info("OpenCV barcode detector initialized")
            except Exception as e:
                self.logger.error(f"Could not initialize barcode detector: {e}")
                self.barcode_detector = None
        else:
            self.barcode_detector = None
        
        # Initialize QR code detector
        try:
            self.qr_detector = cv2.QRCodeDetector()
            self.logger.info("OpenCV QR detector initialized")
        except Exception as e:
            self.logger.error(f"Could not initialize QR detector: {e}")
            self.qr_detector = None
        
        # Lazy load OCR to avoid startup time
        self.ocr_reader = None
        
    def _init_ocr(self):
        """Lazy initialize OCR reader."""
        if self.ocr_reader is None and self.use_ocr:
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=self.use_gpu)
                self.logger.info(f"EasyOCR initialized with languages: {self.ocr_languages}")
            except Exception as e:
                self.logger.warning(f"Could not initialize OCR: {e}")
                self.use_ocr = False
    
    def detect_codes(self, image: np.ndarray) -> Dict[str, List[str]]:
        """
        Detect barcodes and QR codes in an image using OpenCV.
        
        Args:
            image: Input image (can be cropped label)
            
        Returns:
            Dictionary with 'barcodes' and 'qrcodes' lists
        """
        result = {
            'barcodes': [],
            'qrcodes': []
        }
        
        self.logger.info(f"Attempting barcode/QR detection on image shape: {image.shape}, dtype: {image.dtype}")
        
        # Save debug image (first detection only)
        import os
        debug_dir = "logs/debug_crops"
        os.makedirs(debug_dir, exist_ok=True)
        debug_files = len([f for f in os.listdir(debug_dir) if f.startswith('crop_')])
        if debug_files < 3:  # Save first 3 crops for debugging
            debug_path = f"{debug_dir}/crop_{debug_files}.png"
            cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image)
            self.logger.info(f"Saved debug crop to {debug_path}")
        
        # Ensure image is grayscale for better detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize image if too small (OpenCV barcode detector needs reasonable size)
        min_size = 200
        h, w = gray.shape[:2]
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            self.logger.info(f"Upscaled image from {w}x{h} to {new_w}x{new_h}")
        
        # Apply preprocessing to improve detection
        # Try multiple preprocessing approaches
        images_to_try = [gray]
        
        # Try with bilateral filter to reduce noise while keeping edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        images_to_try.append(denoised)
        
        # Try with thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images_to_try.append(thresh)
        
        # Try with adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        images_to_try.append(adaptive)
        
        # Use pyzbar if enabled (much more reliable)
        if self.use_pyzbar:
            try:
                # pyzbar works best with the original color image
                decoded_objects = pyzbar.decode(image)
                self.logger.info(f"pyzbar found {len(decoded_objects)} codes")
                
                for obj in decoded_objects:
                    code_data = obj.data.decode('utf-8')
                    code_type = obj.type
                    
                    if code_type == 'QRCODE':
                        if code_data not in result['qrcodes']:
                            result['qrcodes'].append(code_data)
                            self.logger.info(f"Found QR code: {code_data}")
                    else:
                        if code_data not in result['barcodes']:
                            result['barcodes'].append(code_data)
                            self.logger.info(f"Found {code_type} barcode: {code_data}")
                
                # If pyzbar found codes, we're done
                if result['barcodes'] or result['qrcodes']:
                    return result
                    
            except Exception as e:
                self.logger.error(f"Error with pyzbar detection: {e}")
        
        # Detect barcodes with OpenCV (fallback or if pyzbar not enabled)
        if self.barcode_detector is not None:
            for idx, img in enumerate(images_to_try):
                try:
                    # OpenCV BarcodeDetector.detectAndDecode returns (retval, decoded_info, points)
                    retval, decoded_info, points = self.barcode_detector.detectAndDecode(img)
                    self.logger.info(f"Barcode detection attempt {idx+1}: retval={retval}, decoded_info='{decoded_info}'")
                    
                    if retval and decoded_info:
                        # Successfully found barcode
                        if isinstance(decoded_info, (list, tuple)):
                            for code in decoded_info:
                                if code and code not in result['barcodes']:
                                    result['barcodes'].append(code)
                                    self.logger.info(f"Found barcode: {code}")
                        else:
                            if decoded_info and decoded_info not in result['barcodes']:
                                result['barcodes'].append(decoded_info)
                                self.logger.info(f"Found barcode: {decoded_info}")
                        
                        # If we found barcodes, no need to try other preprocessing
                        if result['barcodes']:
                            break
                except Exception as e:
                    self.logger.error(f"Error detecting barcodes (attempt {idx+1}): {e}")
            
            if not result['barcodes']:
                self.logger.info("No barcodes detected after trying multiple preprocessing methods")
        
        # Detect QR codes
        if self.qr_detector is not None:
            for idx, img in enumerate(images_to_try):
                try:
                    retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(img)
                    self.logger.info(f"QR detection attempt {idx+1}: retval={retval}, codes={len(decoded_info) if decoded_info else 0}")
                    
                    if retval and decoded_info:
                        for code in decoded_info:
                            if code and code not in result['qrcodes']:
                                result['qrcodes'].append(code)
                                self.logger.info(f"Found QR code: {code}")
                        
                        # If we found QR codes, no need to try other preprocessing
                        if result['qrcodes']:
                            break
                except Exception as e:
                    self.logger.error(f"Error detecting QR codes (attempt {idx+1}): {e}")
            
            if not result['qrcodes']:
                self.logger.info("No QR codes detected after trying multiple preprocessing methods")
        
        return result
    
    def read_text_ocr(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Optional[str]:
        """
        Extract text from image using OCR.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            Extracted text or None if no text found
        """
        # Initialize OCR if needed
        if self.ocr_reader is None:
            self._init_ocr()
        
        if self.ocr_reader is None:
            return None
        
        try:
            # Run OCR
            results = self.ocr_reader.readtext(image)
            
            # Filter by confidence and combine text
            text_parts = []
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    text_parts.append(text)
                    self.logger.debug(f"OCR detected: '{text}' (confidence: {confidence:.2f})")
            
            if text_parts:
                return '\n'.join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Error during OCR: {e}")
        
        return None
    
    def process_label(
        self, 
        image: np.ndarray, 
        try_codes_first: bool = True,
        ocr_fallback: bool = True,
        ocr_confidence: float = 0.5
    ) -> Dict:
        """
        Process a label image through the complete pipeline.
        
        Pipeline:
        1. Try to detect barcodes/QR codes
        2. If no codes found and ocr_fallback=True, run OCR
        
        Args:
            image: Label image crop
            try_codes_first: Try barcode/QR detection first
            ocr_fallback: Use OCR if no codes detected
            ocr_confidence: Minimum OCR confidence threshold
            
        Returns:
            Dictionary with detected data:
            {
                'barcodes': [...],
                'qrcodes': [...],
                'text': '...' or None,
                'detection_method': 'codes'/'ocr'/'both'/'none'
            }
        """
        result = {
            'barcodes': [],
            'qrcodes': [],
            'text': None,
            'detection_method': 'none'
        }
        
        # Step 1: Try code detection
        if try_codes_first:
            codes = self.detect_codes(image)
            result['barcodes'] = codes['barcodes']
            result['qrcodes'] = codes['qrcodes']
            
            if codes['barcodes'] or codes['qrcodes']:
                result['detection_method'] = 'codes'
        
        # Step 2: OCR fallback or supplement
        if ocr_fallback:
            # Use OCR if no codes found
            if not result['barcodes'] and not result['qrcodes']:
                text = self.read_text_ocr(image, ocr_confidence)
                if text:
                    result['text'] = text
                    result['detection_method'] = 'ocr'
            else:
                # Codes found, optionally also get text
                text = self.read_text_ocr(image, ocr_confidence)
                if text:
                    result['text'] = text
                    result['detection_method'] = 'both'
        
        return result
    
    def process_labels_batch(
        self, 
        label_images: List[np.ndarray],
        **kwargs
    ) -> List[Dict]:
        """
        Process multiple label images.
        
        Args:
            label_images: List of label image crops
            **kwargs: Additional arguments for process_label()
            
        Returns:
            List of detection results
        """
        results = []
        for i, image in enumerate(label_images):
            try:
                result = self.process_label(image, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing label {i}: {e}")
                results.append({
                    'barcodes': [],
                    'qrcodes': [],
                    'text': None,
                    'detection_method': 'error',
                    'error': str(e)
                })
        
        return results
