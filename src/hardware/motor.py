import time
import RPi.GPIO as GPIO

# GPIO pins: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#gpio


class MotorController:

    def __init__(self):
        self.pins: tuple[int, int, int, int] = (26, 19, 13, 6)
        self.step_sequence: list[list[int]] = [[1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0],
                                               [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
        self.steps_per_rev: int = 4096
        self.llama_rotate_degrees: int = 15
        self.duck_rotate_degrees: int = 15
        self.step_delay_secs: float = 0.002
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the GPIO pins for motor control.

        Returns:

        """
        GPIO.setmode(GPIO.BCM)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)
        print("Successfully initialized motor.")

    def press_llama(self) -> None:
        """
        Press the llama button after inference.
        Rotate back to original position.

        Returns:

        """
        print("Press llama.")
        steps_to_rotate = (self.steps_per_rev * self.llama_rotate_degrees) // 360
        self.rotate(steps_to_rotate, 1)
        time.sleep(0.1)
        self.rotate(steps_to_rotate, -1)

    def press_duck(self) -> None:
        """
        Press the duck button after inference.
        Rotate back to original position.

        Returns:

        """
        print("Press duck.")
        steps_to_rotate = (self.steps_per_rev * self.duck_rotate_degrees) // 360
        self.rotate(steps_to_rotate, -1)
        time.sleep(0.1)
        self.rotate(steps_to_rotate, 1)

    def rotate(self, num_steps: int, direction: int) -> None:
        """
        Rotate the motor.

        Args:
            num_steps: Number of full steps to make.
            direction: The direction to rotate.

        Returns:

        """
        semi_step_counter: int = 0
        for step in range(num_steps):
            if direction == 1:
                self.clockwise_step(semi_step_counter)
            else:
                self.counter_clockwise_step(semi_step_counter)
            semi_step_counter = (semi_step_counter + 1) % 8
            time.sleep(self.step_delay_secs)

    def clockwise_step(self, semi_step: int) -> None:
        """
        Rotate the motor clockwise.

        Args:
            semi_step: The semi-step can be 1-8.

        Returns:

        """
        GPIO.output(self.pins[0], self.step_sequence[semi_step][0])
        GPIO.output(self.pins[1], self.step_sequence[semi_step][1])
        GPIO.output(self.pins[2], self.step_sequence[semi_step][2])
        GPIO.output(self.pins[3], self.step_sequence[semi_step][3])

    def counter_clockwise_step(self, semi_step: int) -> None:
        """
        Rotate the motor counter-clockwise.

        Args:
            semi_step: The semi-step can be 1-8.

        Returns:

        """
        GPIO.output(self.pins[0], self.step_sequence[semi_step][3])
        GPIO.output(self.pins[1], self.step_sequence[semi_step][2])
        GPIO.output(self.pins[2], self.step_sequence[semi_step][1])
        GPIO.output(self.pins[3], self.step_sequence[semi_step][0])

    def cleanup_motor(self) -> None:
        """
        Clean up the GPIO pins when done using.

        Returns:

        """
        for pin in self.pins:
            GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup()
        print("Cleaned up motor")


if __name__ == "__main__":
    controller = MotorController()
    controller.initialize()
    start_time = time.time()
    for _ in range(10):
        controller.press_llama()
        controller.press_duck()
    end_time = time.time()
    print(f"{(10 / (end_time - start_time)) * 2} moves/second")
    controller.cleanup_motor()
