import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torchvision.transforms as T

from model import MiniCLIP
from config import Config


class MiniCLIPInference:
    def __init__(self, weights_path: str = "mini_vlm_best.pth", device=None):
        self.device = device or Config.DEVICE

        self.model = MiniCLIP().to(self.device)
        self._load_weights(weights_path)
        self.model.eval()

        self.tokenizer = self._load_tokenizer()
        self.transform = self._build_transform()

    def _load_weights(self, weights_path: str) -> None:
        checkpoint = torch.load(weights_path, map_location=self.device)

        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
            print(
                f"Loaded weights from epoch {checkpoint['epoch']} "
                f"(loss: {checkpoint['loss']:.4f})"
            )
        else:
            self.model.load_state_dict(checkpoint)
            print("Loaded raw state_dict")

    def _load_tokenizer(self) -> ByteLevelBPETokenizer:
        tokenizer = ByteLevelBPETokenizer(
            "flickr_bpe-vocab.json",
            "flickr_bpe-merges.txt",
        )

        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )

        tokenizer.enable_truncation(max_length=Config.MAX_LEN)
        tokenizer.enable_padding(
            length=Config.MAX_LEN,
            pad_id=tokenizer.token_to_id("<pad>"),
        )

        return tokenizer

    def _build_transform(self) -> T.Compose:
        return T.Compose(
            [
                T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, image_path: str, possible_captions: list[str]) -> None:
        image = self._load_image(image_path)
        if image is None:
            return

        image_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        encoded = self.tokenizer.encode_batch(possible_captions)
        input_ids = torch.tensor([e.ids for e in encoded]).to(self.device)
        attention_mask = torch.tensor(
            [e.attention_mask for e in encoded]
        ).to(self.device)

        with torch.no_grad():
            image_embedding, text_embeddings = self.model(
                image_tensor, input_ids, attention_mask
            )

            logits = (
                image_embedding @ text_embeddings.T
            ) * self.model.logit_scale.exp()

            probabilities = (
                F.softmax(logits, dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        self._plot_prediction(image, possible_captions, probabilities)

    def _load_image(self, image_path: str):
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            print(f"Failed to load image: {image_path}")
            return None

    def _plot_prediction(
        self,
        image: Image.Image,
        captions: list[str],
        probabilities: np.ndarray,
    ) -> None:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Query Image")

        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(captions))
        colors = [
            "green" if i == np.argmax(probabilities) else "gray"
            for i in range(len(captions))
        ]

        plt.barh(y_pos, probabilities, align="center", color=colors)
        plt.yticks(y_pos, captions)
        plt.xlabel("Confidence Score")
        plt.title("Zero-Shot Prediction")
        plt.xlim(0.0, 1.0)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print(
        "MiniCLIPInference loaded. "
        "Import and use this class from a notebook or service."
    )