
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
from pathlib import Path

from PIL import Image
from ppdiffusers.transformers.trocr import TrOCRProcessor
from ppdiffusers.transformers.vision_encoder_decoder import VisionEncoderDecoderModel

img_path = "dataset/UniMER-Test/spe/0000035.png"
img = Image.open(img_path).convert("RGB")
img_stem = Path(img_path).stem

s1 = time.perf_counter()
print("Loading model")
model_path = "./trocr-paddle"
processor = TrOCRProcessor.from_pretrained("./trocr-small-stage1")
pixel_values = processor(img, return_tensors="pd").pixel_values
model = VisionEncoderDecoderModel.from_pretrained(model_path)
print("Finished loading model.")
s2 = time.perf_counter()

# return_pred_ids is for paddle, to directly return pred_ids.
generated_ids = model.generate(pixel_values, return_pred_ids = True)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=None)[0]
s3 = time.perf_counter()

print(generated_text)
print(f"loading_model: {s2 - s1}s")
print(f"infer: {s3 - s2}s")
