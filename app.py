import torch
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

from inference import MiniCLIPInference
from config import Config

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_mac_device():
    if torch.backends.mps.is_available():
        print("Apple Silicon Detected: Using Metal acceleration.")
        return torch.device("mps")
    else:
        print("Apple Silicon not detected: Using standard CPU.")
        return torch.device("cpu")

def run_demo():
    
    device = get_mac_device()
    
    print("\nLoading Model... (This might take a moment)")
    try:
        infer = MiniCLIPInference(weights_path="mini_vlm_best.pth", device=device)
        print("Model Loaded Successfully!")
    except FileNotFoundError:
        print("Error: 'mini_vlm_best.pth' not found in this folder.")
        return
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        return

    while True:
        print("\n" + "=" * 40)
        image_path = input("Drag and drop an image file here (or type 'q' to quit): ").strip()
        
        image_path = image_path.replace("'", "").replace('"', "").strip()
        
        if image_path.lower() == 'q':
            break
            
        if not os.path.exists(image_path):
            print("File does not exist. Try again.")
            continue
            
        print("\nEnter 3 options describing the image:")
        c1 = input("   Option 1: ").strip()
        c2 = input("   Option 2: ").strip()
        c3 = input("   Option 3: ").strip()
        
        if not (c1 and c2 and c3):
            print("Please enter all 3 options.")
            continue
            
        captions = [c1, c2, c3]
        
        print("\nThinking...")
        try:
            infer.predict(image_path, captions)
            print("Prediction displayed in popup window.")
        except Exception as e:
            print(f"Error during inference: {e}")

if __name__ == "__main__":
    run_demo()