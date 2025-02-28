import time
from PIL import Image
from picamera2 import Picamera2, Preview


class CameraController:

    def __init__(self) -> None:
        self.camera: Picamera2 = Picamera2()
        still_config = self.camera.create_still_configuration(main={"size": (580, 680), "format": "BGR888"})
        self.camera.configure(still_config)
        self.camera.start_preview(Preview.NULL)
        self.camera.start()
        print("\nSuccessfully initialized camera")
        time.sleep(1)

    def take_image(self) -> Image:
        """
        Take a single image with camera.

        Returns:
            A PIL Image that is RGB 8-bit.
        """
        pil_image = self.camera.capture_image()
        return pil_image

    def stop_camera(self) -> None:
        """
        Stop the camera when done using.

        Returns:

        """
        print('Stopping camera')
        self.camera.stop()


if __name__ == "__main__":
    controller = CameraController()
    start_time = time.time()
    for _ in range(10):
        _ = controller.take_image()
    end_time = time.time()
    print(f"{10 / (end_time - start_time)} images/second")
    controller.stop_camera()
