import torch

class Config:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    IMAGE_DIR = (
        "/kaggle/input/flickr-image-dataset/"
        "flickr30k_images/flickr30k_images/"
    )

    CAPTIONS_FILE = (
        "/kaggle/input/flickr-image-dataset/"
        "flickr30k_images/results.csv"
    )

    IMG_SIZE = 128
    PATCH_SIZE = 16
    MAX_LEN = 64
    BATCH_SIZE = 256
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    EMBED_DIM = 256
    VOCAB_SIZE = 10_000
    NUM_HEADS = 4
    NUM_LAYERS = 4
    DROPOUT = 0.1