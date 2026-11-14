# Inference-only image for the Gradio demo (CPU).
# Before building: python scripts/package_demo.py --run runs/<best run>
#   docker build -t polyp-demo .
#   docker run --rm -p 7860:7860 polyp-demo
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "segmentation-models-pytorch>=0.5" "opencv-python-headless>=4.9" "gradio>=5.0" \
    "albumentations>=2.0"

COPY src ./src
COPY app ./app
COPY checkpoints/model.pt ./checkpoints/model.pt

ENV HOST=0.0.0.0 PORT=7860 NO_ALBUMENTATIONS_UPDATE=1
EXPOSE 7860
CMD ["python", "app/app.py"]
