from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw, UnidentifiedImageError
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TableDetector:
    def __init__(self, model_name:str, detection_threshold:float = 0.8, device:str | None = None):
        self.model_name = model_name
        self.detection_threshold = detection_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()       # inference mode

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from a file path and convert to RGB."""
        try:
            image = Image.open(image_path)
            return image.convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except OSError:
            raise ValueError(f"Cannot open image file: {image_path}")

    def _forward(self, image: Image.Image) -> dict:
        """Perform a forward pass through the model."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def _post_process(self, outputs: dict, image_size: tuple) -> dict:
        """Post-process the model outputs to extract bounding boxes, labels, and scores."""
        target_sizes = torch.tensor(
            [image_size[::-1]],
            device=self.device
            )
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.detection_threshold
        )[0]
        return {
            "boxes": results["boxes"],
            "labels": results["labels"],
            "scores": results["scores"]
        }
    
    def predict(self, image_path: str) -> dict:
        """
        Predict tables in the given image.
        
        Args:
            image_path (str): Path to the input image file. The image should be a valid document image (e.g., invoice or bank statement).

        Returns:
            dict: A dictionary containing detection results:
                - "boxes": list of bounding boxes [x_min, y_min, x_max, y_max]
                - "labels": list of detected object labels (integer)
                - "scores": list of confidence scores (float)
        """
        image = self._load_image(image_path)
        outputs = self._forward(image)
        results =self._post_process(outputs, image.size)

        if len(results["boxes"]) == 0:
            print("No tables detected.")
        else:
            print(f"Tables detected: {results['boxes'].shape[0]}")
        return results

    def visualize(self, image_path: str, results:dict, output_path: str = None, show: bool = True) -> Image.Image:
        """
        Draw detected table boxes on the image.
        
        Args:
            - image_path (str): Path to the input image file.
            - results (dict): Detection results from `predict()` containing "boxes", "labels", and "scores".
            - output_path (str, optional): Path to save the visualized image. If None, the image is not saved.
            - show (bool, optional): Whether to display the image after drawing boxes. Default is True.

        Returns:
            - PIL.Image.Image: Image object with bounding boxes drawn.
        """
        # load image
        image = self._load_image(image_path)
        draw = ImageDraw.Draw(image)

        # draw boxes
        for box in results["boxes"]:
            x_min, y_min, x_max, y_max = box.tolist()
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)

        # display
        if show:
            image.show()

        # save
        if output_path:
            try:
                image.save(output_path)
            except Exception as e:
                print(f"Error saving image: {e}")
        return image


