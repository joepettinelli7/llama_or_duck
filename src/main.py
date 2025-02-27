import torch

from hardware.motor import MotorController
from hardware.camera import CameraController
from modeling.inference.inference_maker import InferenceMaker


def play_game() -> None:
    """
    Play llama or duck until keyboard interrupt.
    """
    camera = CameraController()
    motor = MotorController()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inference_maker = InferenceMaker(device)
    try:
        while True:
            img = camera.take_image()
            is_llama = inference_maker.infer(img)
            if is_llama:
                motor.press_llama()
            else:
                motor.press_duck()
    except KeyboardInterrupt:
        camera.stop_camera()
        motor.cleanup_motor()


if __name__ == "__main__":
    play_game()
