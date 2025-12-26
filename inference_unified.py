#!/usr/bin/env python3
"""
Unified inference script for WASB (TrackNetV2) and VBallNet models.
Supports inference, visualization, and accuracy evaluation using Mean Euclidean Distance Error.

Usage examples:
    # Using uv with dependencies from fast-volleyball-tracking-inference/uv.lock
    uv run inference_unified.py --model vballnet --test_input_path video.mp4 --output_path ./output
    
    # Or with traditional pip + python
    python inference_unified.py --model wasb --test_input_path video.mp4 --visualize --output_path ./output
    
    # Both models with accuracy calculation on OREL dataset
    uv run inference_unified.py --model wasb vballnet --test_input_path video.mp4 \\
        --accuracy --dataset_orel --annotations_path annotations.csv --output_path ./output
    
    # Accuracy calculation on frames with deep activity rec dataset
    uv run inference_unified.py --model vballnet --test_input_path ./frames_dir \\
        --accuracy --dataset_deep_activity_rec --annotations_path annotations.txt --output_path ./output
"""

import argparse
import os
import sys
import json
import csv
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import deque
from typing import List, Dict, Tuple, Optional
import onnxruntime as ort
from tqdm import tqdm

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent

# Add paths for WASB-SBDT imports
wasb_src_path = SCRIPT_DIR / 'WASB-SBDT' / 'src'
if wasb_src_path.exists():
    sys.path.insert(0, str(wasb_src_path))


class WASBInference:
    """TrackNetV2 model inference wrapper for WASB."""
    
    def __init__(self, weights_path: str, device: str = 'cuda', score_threshold: float = 0.5):
        from models.unet2d import TrackNetV2
        
        self.device = device
        self.score_threshold = score_threshold
        self.frames_in = 3
        self.inp_width = 512
        self.inp_height = 288
        
        # Build model
        self.model = TrackNetV2(
            n_channels=self.frames_in * 3,
            n_classes=self.frames_in,
            bilinear=True,
            halve_channel=False
        )
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(device)
        self.model.eval()
        
        # Image transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame."""
        frame_resized = cv2.resize(frame, (self.inp_width, self.inp_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame_rgb)
        return frame_tensor
    
    def detect_ball_from_heatmap(self, heatmap: np.ndarray) -> List[Dict]:
        """Extract ball detections from heatmap."""
        detections = []
        
        if np.max(heatmap) <= self.score_threshold:
            return detections
        
        _, hm_th = cv2.threshold(heatmap, self.score_threshold, 1, cv2.THRESH_BINARY)
        n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
        
        for m in range(1, n_labels):
            ys, xs = np.where(labels == m)
            ws = heatmap[ys, xs]
            
            score = float(ws.sum())
            x = float(np.sum(xs * ws) / np.sum(ws))
            y = float(np.sum(ys * ws) / np.sum(ws))
            
            detections.append({
                'x': x,
                'y': y,
                'score': score
            })
        
        return detections
    
    def scale_detections(self, detections: List[Dict], orig_width: int, orig_height: int) -> List[Dict]:
        """Scale detections from model output size to original frame size."""
        scale_x = orig_width / self.inp_width
        scale_y = orig_height / self.inp_height
        
        scaled = []
        for det in detections:
            scaled.append({
                'x': det['x'] * scale_x,
                'y': det['y'] * scale_y,
                'score': det['score']
            })
        return scaled
    
    @torch.no_grad()
    def process_video(self, video_path: str) -> List[Dict]:
        """Process video and return detections for each frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"WASB: Processing video {Path(video_path).name}")
        print(f"  Resolution: {orig_width}x{orig_height}, FPS: {fps:.2f}, Frames: {total_frames}")
        
        results = []
        frame_buffer = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="WASB inference", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_tensor = self.preprocess_frame(frame)
            frame_buffer.append(frame_tensor)
            
            if len(frame_buffer) >= self.frames_in:
                input_tensor = torch.cat(frame_buffer[-self.frames_in:], dim=0)
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
                
                output = self.model(input_tensor)
                heatmaps = output[0].sigmoid().cpu().numpy()[0]
                
                for i in range(self.frames_in):
                    result_frame_idx = frame_idx - self.frames_in + 1 + i
                    
                    if result_frame_idx < 0 or any(r['frame'] == result_frame_idx for r in results):
                        continue
                    
                    hm = heatmaps[i]
                    detections = self.detect_ball_from_heatmap(hm)
                    detections = self.scale_detections(detections, orig_width, orig_height)
                    
                    if detections:
                        # Take the detection with highest score
                        det = max(detections, key=lambda x: x['score'])
                        results.append({
                            'frame': result_frame_idx,
                            'visibility': 1,
                            'x': det['x'],
                            'y': det['y']
                        })
                    else:
                        results.append({
                            'frame': result_frame_idx,
                            'visibility': 0,
                            'x': -1,
                            'y': -1
                        })
                
                if len(frame_buffer) > self.frames_in:
                    frame_buffer = frame_buffer[-(self.frames_in-1):]
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        results.sort(key=lambda x: x['frame'])
        return results
    
    @torch.no_grad()
    def process_frames(self, frames_dir: str) -> List[Dict]:
        """Process directory of frames and return detections."""
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        if not frame_files:
            raise ValueError(f"No image files found in {frames_dir}")
        
        print(f"WASB: Processing {len(frame_files)} frames from {frames_dir}")
        
        results = []
        frame_buffer = []
        
        pbar = tqdm(total=len(frame_files), desc="WASB inference", unit="frame")
        
        for frame_idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            frame_tensor = self.preprocess_frame(frame)
            frame_buffer.append(frame_tensor)
            
            if len(frame_buffer) >= self.frames_in:
                input_tensor = torch.cat(frame_buffer[-self.frames_in:], dim=0)
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
                
                output = self.model(input_tensor)
                heatmaps = output[0].sigmoid().cpu().numpy()[0]
                
                for i in range(self.frames_in):
                    result_frame_idx = frame_idx - self.frames_in + 1 + i
                    
                    if result_frame_idx < 0 or any(r['frame'] == result_frame_idx for r in results):
                        continue
                    
                    hm = heatmaps[i]
                    detections = self.detect_ball_from_heatmap(hm)
                    if detections:
                        orig_size = frame.shape[:2]
                        detections = self.scale_detections(detections, orig_size[1], orig_size[0])
                        det = max(detections, key=lambda x: x['score'])
                        results.append({
                            'frame': result_frame_idx,
                            'visibility': 1,
                            'x': det['x'],
                            'y': det['y']
                        })
                    else:
                        results.append({
                            'frame': result_frame_idx,
                            'visibility': 0,
                            'x': -1,
                            'y': -1
                        })
                
                if len(frame_buffer) > self.frames_in:
                    frame_buffer = frame_buffer[-(self.frames_in-1):]
            
            pbar.update(1)
        
        pbar.close()
        results.sort(key=lambda x: x['frame'])
        return results


class VBallNetInference:
    """VBallNet ONNX model inference wrapper."""
    
    def __init__(self, model_path: str, score_threshold: float = 0.5):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        self.score_threshold = score_threshold
        self.inp_width = 512
        self.inp_height = 288
        
        # Determine sequence length from model filename
        if "seq15" in model_path.lower():
            self.seq_length = 15
        elif "seq9" in model_path.lower():
            self.seq_length = 9
        else:
            self.seq_length = 3
        
        print(f"VBallNet: Model loaded, sequence length: {self.seq_length}")
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Preprocess frames to grayscale and resize."""
        processed = []
        for frame in frames:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame_gray, (self.inp_width, self.inp_height))
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            processed.append(frame_normalized)
        return np.array(processed)
    
    def postprocess_heatmap(self, heatmap: np.ndarray) -> Tuple[int, float, float]:
        """Extract ball position from heatmap."""
        _, binary = cv2.threshold(heatmap, self.score_threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                return 1, cx, cy
        
        return 0, 0.0, 0.0
    
    def process_video(self, video_path: str) -> List[Dict]:
        """Process video and return detections for each frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"VBallNet: Processing video {Path(video_path).name}")
        print(f"  Resolution: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}")
        
        results = []
        frame_buffer = deque(maxlen=self.seq_length)
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="VBallNet inference", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_buffer.append(frame)
            
            if len(frame_buffer) == self.seq_length:
                frames_array = np.array(list(frame_buffer))
                processed_frames = self.preprocess_frames(frames_array)
                
                # Prepare input: (1, seq_len, height, width)
                # processed_frames already has shape (seq_len, height, width)
                input_tensor = np.expand_dims(processed_frames, axis=0)
                
                # Run inference
                input_name = self.session.get_inputs()[0].name
                output = self.session.run(None, {input_name: input_tensor})[0]
                
                # Process heatmaps for all frames in sequence
                for i in range(self.seq_length):
                    heatmap = output[0, i, :, :]
                    visibility, x, y = self.postprocess_heatmap(heatmap)
                    
                    # Scale to original resolution
                    if visibility:
                        x_orig = x * frame_width / self.inp_width
                        y_orig = y * frame_height / self.inp_height
                    else:
                        x_orig = -1
                        y_orig = -1
                    
                    result_frame_idx = frame_idx - self.seq_length + 1 + i
                    if result_frame_idx >= 0:
                        results.append({
                            'frame': result_frame_idx,
                            'visibility': visibility,
                            'x': x_orig,
                            'y': y_orig
                        })
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        results.sort(key=lambda x: x['frame'])
        return results
    
    def process_frames(self, frames_dir: str) -> List[Dict]:
        """Process directory of frames and return detections."""
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        if not frame_files:
            raise ValueError(f"No image files found in {frames_dir}")
        
        print(f"VBallNet: Processing {len(frame_files)} frames from {frames_dir}")
        
        results = []
        frame_buffer = deque(maxlen=self.seq_length)
        
        pbar = tqdm(total=len(frame_files), desc="VBallNet inference", unit="frame")
        
        for frame_idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            frame_buffer.append(frame)
            
            if len(frame_buffer) == self.seq_length:
                frames_array = np.array(list(frame_buffer))
                processed_frames = self.preprocess_frames(frames_array)
                
                # Prepare input: (1, seq_len, height, width)
                # processed_frames already has shape (seq_len, height, width)
                input_tensor = np.expand_dims(processed_frames, axis=0)
                
                input_name = self.session.get_inputs()[0].name
                output = self.session.run(None, {input_name: input_tensor})[0]
                
                frame_height, frame_width = frame.shape[:2]
                
                for i in range(self.seq_length):
                    heatmap = output[0, i, :, :]
                    visibility, x, y = self.postprocess_heatmap(heatmap)
                    
                    if visibility:
                        x_orig = x * frame_width / self.inp_width
                        y_orig = y * frame_height / self.inp_height
                    else:
                        x_orig = -1
                        y_orig = -1
                    
                    result_frame_idx = frame_idx - self.seq_length + 1 + i
                    if result_frame_idx >= 0:
                        results.append({
                            'frame': result_frame_idx,
                            'visibility': visibility,
                            'x': x_orig,
                            'y': y_orig
                        })
            
            pbar.update(1)
        
        pbar.close()
        results.sort(key=lambda x: x['frame'])
        return results


class AccuracyCalculator:
    """Calculate Mean Euclidean Distance Error (MEDE)."""
    
    @staticmethod
    def load_annotations_orel(csv_path: str) -> Dict[int, Tuple[int, float, float]]:
        """Load OREL dataset annotations from CSV."""
        annotations = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row['Frame'])
                visibility = int(row['Visibility'])
                x = float(row['X'])
                y = float(row['Y'])
                annotations[frame] = (visibility, x, y)
        return annotations
    
    @staticmethod
    def load_annotations_deep_activity_rec(txt_path: str) -> Dict[int, Tuple[int, float, float]]:
        """Load deep activity rec dataset annotations from TXT."""
        annotations = {}
        with open(txt_path, 'r') as f:
            for frame_idx, line in enumerate(f):
                line = line.strip()
                if line:
                    parts = line.split()
                    x = float(parts[0])
                    y = float(parts[1])
                    annotations[frame_idx] = (1, x, y)
                else:
                    annotations[frame_idx] = (0, -1, -1)
        return annotations
    
    @staticmethod
    def calculate_mede(predictions: List[Dict], annotations: Dict) -> Dict:
        """Calculate Mean Euclidean Distance Error."""
        distances = []
        matched_frame_indices = set()
        total_annotated = len(annotations)
        
        # Deduplicate predictions by frame (for seq models that output same frame multiple times)
        # Keep only first occurrence of each frame
        seen_frames = set()
        unique_predictions = []
        for pred in predictions:
            frame_idx = pred['frame']
            if frame_idx not in seen_frames:
                unique_predictions.append(pred)
                seen_frames.add(frame_idx)
        
        for pred in unique_predictions:
            frame_idx = pred['frame']
            if frame_idx not in annotations:
                continue
            
            matched_frame_indices.add(frame_idx)
            pred_visibility = pred['visibility']
            pred_x = pred['x']
            pred_y = pred['y']
            
            ann_visibility, ann_x, ann_y = annotations[frame_idx]
            
            # Only calculate distance if both are visible
            if pred_visibility and ann_visibility and ann_x >= 0 and ann_y >= 0:
                distance = np.sqrt((pred_x - ann_x) ** 2 + (pred_y - ann_y) ** 2)
                distances.append(distance)
        
        if distances:
            mede = np.mean(distances)
        else:
            mede = float('inf')
        
        return {
            'mede': mede,
            'matched_frames': len(matched_frame_indices),
            'total_annotated': total_annotated,
            'frames_with_distance': len(distances)
        }


class Visualizer:
    """Visualize detections on frames."""
    
    @staticmethod
    def draw_detections(frame: np.ndarray, predictions: List[Dict], model_name: str) -> np.ndarray:
        """Draw all predictions on a frame."""
        vis_frame = frame.copy()
        color = (0, 255, 0) if model_name == 'wasb' else (255, 0, 0)
        
        for pred in predictions:
            if pred['visibility'] and pred['x'] >= 0 and pred['y'] >= 0:
                x = int(pred['x'])
                y = int(pred['y'])
                cv2.circle(vis_frame, (x, y), 8, color, -1)
                cv2.putText(vis_frame, model_name, (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    @staticmethod
    def create_visualization_video(
        video_path: str,
        predictions_dict: Dict[str, List[Dict]],
        output_path: str,
        output_dir: str,
        dataset_type: str = None
    ):
        """Create visualization video with predictions."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        is_video = True
        
        # Create output directories
        for model_name in predictions_dict.keys():
            if dataset_type:
                model_video_path, _ = get_output_paths(video_path, output_path, dataset_type, is_video, model_name=model_name)
                model_output_dir = os.path.dirname(model_video_path)
                output_video_path = model_video_path
            else:
                model_output_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)
                output_video_path = os.path.join(model_output_dir, 'visualization.mp4')
            
            out_writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            
            print(f"Creating visualization video for {model_name}...")
            pbar = tqdm(total=total_frames, desc=f"Visualizing {model_name}", unit="frame")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get predictions for this frame
                frame_preds = [p for p in predictions_dict[model_name] if p['frame'] == frame_idx]
                
                # Draw detections
                vis_frame = Visualizer.draw_detections(frame, frame_preds, model_name)
                out_writer.write(vis_frame)
                
                frame_idx += 1
                pbar.update(1)
            
            pbar.close()
            out_writer.release()
            print(f"  Visualization saved to: {output_video_path}")
        
        cap.release()
    
    @staticmethod
    def create_visualization_from_frames(
        frames_dir: str,
        predictions_dict: Dict[str, List[Dict]],
        output_dir: str,
        dataset_type: str = None
    ):
        """Create visualization video from frame directory."""
        # Get frame files
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        if not frame_files:
            print(f"No frames found in {frames_dir}")
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        if first_frame is None:
            print(f"Cannot read first frame: {frame_files[0]}")
            return
        
        frame_height, frame_width = first_frame.shape[:2]
        fps = 30  # Default FPS for frame sequences
        is_video = False
        
        # Create visualization for each model
        for model_name in predictions_dict.keys():
            if dataset_type:
                model_video_path, _ = get_output_paths(frames_dir, output_dir, dataset_type, is_video, model_name=model_name)
                output_video_path = model_video_path
            else:
                model_output_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)
                output_video_path = os.path.join(model_output_dir, 'visualization.mp4')
            
            out_writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )
            
            print(f"Creating visualization video for {model_name}...")
            pbar = tqdm(total=len(frame_files), desc=f"Visualizing {model_name}", unit="frame")
            
            for frame_idx, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    pbar.update(1)
                    continue
                
                # Get predictions for this frame
                frame_preds = [p for p in predictions_dict[model_name] if p['frame'] == frame_idx]
                
                # Draw detections
                vis_frame = Visualizer.draw_detections(frame, frame_preds, model_name)
                out_writer.write(vis_frame)
                
                pbar.update(1)
            
            pbar.close()
            out_writer.release()
            print(f"  Visualization saved to: {output_video_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified inference script for WASB and VBallNet models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        nargs='+',
        choices=['wasb', 'vballnet'],
        default=['vballnet'],
        help='Model(s) to use for inference'
    )
    
    parser.add_argument(
        '--wasb_weights_path',
        type=str,
        default='WASB-SBDT/pretrained_weights/tracknetv2_volleyball_best.pth.tar',
        help='Path to WASB model weights'
    )
    
    # Input data
    parser.add_argument(
        '--test_input_path',
        type=str,
        required=True,
        help='Path to input video or frames directory'
    )
    
    # Output
    parser.add_argument(
        '--output_path',
        type=str,
        default='./inference_output',
        help='Output directory for results'
    )
    
    # Visualization
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization video with detections'
    )
    
    # Accuracy evaluation
    parser.add_argument(
        '--accuracy',
        action='store_true',
        help='Calculate accuracy metrics'
    )
    
    parser.add_argument(
        '--dataset_orel',
        action='store_true',
        help='Use OREL dataset format'
    )
    
    parser.add_argument(
        '--dataset_deep_activity_rec',
        action='store_true',
        help='Use deep activity rec dataset format'
    )
    
    parser.add_argument(
        '--annotations_path',
        type=str,
        help='Path to annotations file'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate argument combinations."""
    if args.accuracy:
        if not args.annotations_path:
            raise ValueError("--annotations_path is required when --accuracy is enabled")
        
        if args.dataset_orel and args.dataset_deep_activity_rec:
            raise ValueError("--dataset_orel and --dataset_deep_activity_rec are mutually exclusive")
        
        if not (args.dataset_orel or args.dataset_deep_activity_rec):
            raise ValueError("Must specify either --dataset_orel or --dataset_deep_activity_rec with --accuracy")
    
    if not os.path.exists(args.test_input_path):
        raise ValueError(f"Input path does not exist: {args.test_input_path}")
    
    return True


def get_output_paths(input_path: str, output_path: str, dataset_type: str, is_video: bool, model_name: str = None) -> Tuple[str, str]:
    """Get output paths based on dataset type and input path.
    
    For deep_activity_rec:
        video: {output_path}/{model_name}/{parent_folder}/{folder_name}.mp4
        csv: {output_path}/{model_name}/{parent_folder}/{folder_name}.csv
    
    For OREL:
        video: {output_path}/{model_name}/{parent_folder}/video/{video_name}_vis.mp4
        csv: {output_path}/{model_name}/{parent_folder}/csv/{video_name}_ball.csv
    """
    input_path = str(input_path)
    
    if dataset_type == 'deep_activity_rec':
        # For frames directory: extract parent_folder and folder_name
        if not is_video:
            path_parts = Path(input_path).parts
            folder_name = path_parts[-1]  # Last folder
            parent_folder = path_parts[-2]  # Parent folder
            
            if model_name:
                video_output_dir = os.path.join(output_path, model_name, parent_folder)
            else:
                video_output_dir = os.path.join(output_path, parent_folder)
            os.makedirs(video_output_dir, exist_ok=True)
            
            video_file = os.path.join(video_output_dir, f"{folder_name}.mp4")
            csv_file = os.path.join(video_output_dir, f"{folder_name}.csv")
        else:
            # For video file
            path_parts = Path(input_path).parts
            parent_folder = path_parts[-2]  # Parent of video file
            file_name = Path(input_path).stem
            
            if model_name:
                video_output_dir = os.path.join(output_path, model_name, parent_folder)
            else:
                video_output_dir = os.path.join(output_path, parent_folder)
            os.makedirs(video_output_dir, exist_ok=True)
            
            video_file = os.path.join(video_output_dir, f"{file_name}.mp4")
            csv_file = os.path.join(video_output_dir, f"{file_name}.csv")
    
    elif dataset_type == 'orel':
        # For video file: extract parent_folder (dataset name) and video_name
        if is_video:
            path_parts = Path(input_path).parts
            video_name = Path(input_path).stem  # Filename without extension
            parent_folder = path_parts[-3]  # 3 levels up: dataset_name/video/video_file.mp4
            
            if model_name:
                video_output_dir = os.path.join(output_path, model_name, parent_folder, 'video')
                csv_output_dir = os.path.join(output_path, model_name, parent_folder, 'csv')
            else:
                video_output_dir = os.path.join(output_path, parent_folder, 'video')
                csv_output_dir = os.path.join(output_path, parent_folder, 'csv')
            os.makedirs(video_output_dir, exist_ok=True)
            os.makedirs(csv_output_dir, exist_ok=True)
            
            video_file = os.path.join(video_output_dir, f"{video_name}_vis.mp4")
            csv_file = os.path.join(csv_output_dir, f"{video_name}_ball.csv")
        else:
            # For frames directory
            path_parts = Path(input_path).parts
            folder_name = path_parts[-1]
            parent_folder = path_parts[-2]
            
            if model_name:
                video_output_dir = os.path.join(output_path, model_name, parent_folder, 'video')
                csv_output_dir = os.path.join(output_path, model_name, parent_folder, 'csv')
            else:
                video_output_dir = os.path.join(output_path, parent_folder, 'video')
                csv_output_dir = os.path.join(output_path, parent_folder, 'csv')
            os.makedirs(video_output_dir, exist_ok=True)
            os.makedirs(csv_output_dir, exist_ok=True)
            
            video_file = os.path.join(video_output_dir, f"{folder_name}_vis.mp4")
            csv_file = os.path.join(csv_output_dir, f"{folder_name}_ball.csv")
    
    else:
        # Default behavior
        video_file = None
        csv_file = None
    
    return video_file, csv_file


def main():
    args = parse_arguments()
    validate_arguments(args)
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    # Determine if input is video or frames directory
    is_video = os.path.isfile(args.test_input_path) and args.test_input_path.lower().endswith(
        ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    )
    is_frames_dir = os.path.isdir(args.test_input_path)
    
    if not (is_video or is_frames_dir):
        raise ValueError("Input must be a video file or directory of frames")
    
    print(f"Input type: {'Video' if is_video else 'Frames directory'}")
    print(f"Models to run: {', '.join(args.model)}")
    
    # Determine dataset type for output paths
    dataset_type = None
    if args.dataset_orel:
        dataset_type = 'orel'
    elif args.dataset_deep_activity_rec:
        dataset_type = 'deep_activity_rec'
    
    if dataset_type:
        print(f"Dataset type: {dataset_type}")
    print()
    
    # Store predictions for all models
    all_predictions = {}
    
    # Run inference for each model
    if 'wasb' in args.model:
        try:
            wasb = WASBInference(args.wasb_weights_path)
            if is_video:
                predictions = wasb.process_video(args.test_input_path)
            else:
                predictions = wasb.process_frames(args.test_input_path)
            
            all_predictions['wasb'] = predictions
            
            # Save predictions
            if dataset_type:
                wasb_video_path, wasb_csv_path = get_output_paths(
                    args.test_input_path, args.output_path, dataset_type, is_video, model_name='wasb'
                )
            else:
                wasb_output_dir = os.path.join(args.output_path, 'wasb')
                os.makedirs(wasb_output_dir, exist_ok=True)
                wasb_csv_path = os.path.join(wasb_output_dir, 'predictions.csv')
            
            with open(wasb_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Frame', 'Visibility', 'X', 'Y'])
                writer.writeheader()
                for pred in predictions:
                    writer.writerow({
                        'Frame': pred['frame'],
                        'Visibility': pred['visibility'],
                        'X': int(pred['x']) if pred['visibility'] else -1,
                        'Y': int(pred['y']) if pred['visibility'] else -1
                    })
            
            print(f"✓ WASB predictions saved to {wasb_csv_path}\n")
        except Exception as e:
            print(f"✗ WASB inference failed: {e}\n")
    
    if 'vballnet' in args.model:
        try:
            vballnet_path = SCRIPT_DIR / 'fast-volleyball-tracking-inference' / 'models' / 'VballNetFastV1_seq9_grayscale_233_h288_w512.onnx'
            vballnet = VBallNetInference(str(vballnet_path))
            
            if is_video:
                predictions = vballnet.process_video(args.test_input_path)
            else:
                predictions = vballnet.process_frames(args.test_input_path)
            
            all_predictions['vballnet'] = predictions
            
            # Save predictions
            if dataset_type:
                vballnet_video_path, vballnet_csv_path = get_output_paths(
                    args.test_input_path, args.output_path, dataset_type, is_video, model_name='vballnet'
                )
            else:
                vballnet_output_dir = os.path.join(args.output_path, 'vballnet')
                os.makedirs(vballnet_output_dir, exist_ok=True)
                vballnet_csv_path = os.path.join(vballnet_output_dir, 'predictions.csv')
            
            with open(vballnet_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Frame', 'Visibility', 'X', 'Y'])
                writer.writeheader()
                for pred in predictions:
                    writer.writerow({
                        'Frame': pred['frame'],
                        'Visibility': pred['visibility'],
                        'X': int(pred['x']) if pred['visibility'] else -1,
                        'Y': int(pred['y']) if pred['visibility'] else -1
                    })
            
            print(f"✓ VBallNet predictions saved to {vballnet_csv_path}\n")
        except Exception as e:
            print(f"✗ VBallNet inference failed: {e}\n")
    
    # Accuracy calculation
    if args.accuracy and all_predictions:
        print("=" * 60)
        print("ACCURACY EVALUATION")
        print("=" * 60)
        
        # Load annotations
        if args.dataset_orel:
            annotations = AccuracyCalculator.load_annotations_orel(args.annotations_path)
        else:
            annotations = AccuracyCalculator.load_annotations_deep_activity_rec(args.annotations_path)
        
        print(f"Loaded {len(annotations)} annotated frames\n")
        
        # Calculate accuracy for each model
        for model_name, predictions in all_predictions.items():
            metrics = AccuracyCalculator.calculate_mede(predictions, annotations)
            
            print(f"{model_name.upper()}:")
            print(f"  Matched frames: {metrics['matched_frames']}/{metrics['total_annotated']}")
            print(f"  Frames with valid distance: {metrics['frames_with_distance']}")
            if metrics['mede'] != float('inf'):
                print(f"  MEDE: {metrics['mede']:.2f} pixels")
            else:
                print(f"  MEDE: N/A (no valid predictions)")
            print()
    
    # Visualization
    if args.visualize and all_predictions:
        print("=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        if is_video:
            Visualizer.create_visualization_video(
                args.test_input_path,
                all_predictions,
                args.output_path,
                args.output_path,
                dataset_type=dataset_type
            )
        elif is_frames_dir:
            Visualizer.create_visualization_from_frames(
                args.test_input_path,
                all_predictions,
                args.output_path,
                dataset_type=dataset_type
            )
    
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print(f"Results saved to: {args.output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
