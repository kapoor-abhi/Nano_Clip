import onnxruntime as ort
import numpy as np
from PIL import Image
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os

class MacNanoCLIP:
    def __init__(self, model_path="nano_clip_int8.onnx"):
        print(f"Initializing Nano-CLIP Engine ({model_path})")
        

        self.session = ort.InferenceSession(model_path)
        self.tokenizer = ByteLevelBPETokenizer(
            "flickr_bpe-vocab.json", 
            "flickr_bpe-merges.txt"
        )
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_truncation(max_length=64)
        self.tokenizer.enable_padding(length=64, pad_id=self.tokenizer.token_to_id("<pad>"))
        print("Engine Ready.")

    def preprocess_image(self, image_path):
        """
        Mimics PyTorch 'transforms.Normalize' using NumPy
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

 
        img = img.resize((128, 128), Image.Resampling.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, axis=0)
        return img_np

    def predict(self, image_path, captions):

        img_input = self.preprocess_image(image_path)
        if img_input is None: return
        encoded = self.tokenizer.encode_batch(captions)
        input_ids = np.array([enc.ids for enc in encoded], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encoded], dtype=np.float32)

        inputs = {
            "image": img_input,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        outputs = self.session.run(["img_embed", "txt_embed"], inputs)
        img_embeds = outputs[0] 
        txt_embeds = outputs[1] 
        img_embeds /= np.linalg.norm(img_embeds, axis=1, keepdims=True)
        txt_embeds /= np.linalg.norm(txt_embeds, axis=1, keepdims=True)
        logits = np.dot(img_embeds, txt_embeds.T)[0]
        exp_logits = np.exp(logits * 2.65)
        probs = exp_logits / np.sum(exp_logits)

  
        print(f"\nImage: {os.path.basename(image_path)}")
        sorted_indices = np.argsort(probs)[::-1]
        
        for i in sorted_indices:
            score = probs[i] * 100
            bar = "█" * int(score / 5)
            print(f"   {score:.1f}%  {bar}  '{captions[i]}'")

if __name__ == "__main__":
    app = MacNanoCLIP()
    
    print("\nTip: Drag and Drop an image file into this terminal window.")
    
    while True:
        print("\n" + "-"*40)
        img_path = input("Image Path (or 'q' to quit): ").strip().replace("'", "").replace('"', "")
        
        if img_path.lower() == 'q': break
        
        if not os.path.exists(img_path):
            print("File not found. Try again.")
            continue
            
        print("Enter 3 descriptions:")
        c1 = input("   1: ").strip()
        c2 = input("   2: ").strip()
        c3 = input("   3: ").strip()
        
        if c1 and c2 and c3:
            app.predict(img_path, [c1, c2, c3])