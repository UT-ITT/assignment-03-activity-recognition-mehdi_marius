"""simple pyglet window for live activity prediction"""

# simple window that starts prediction from one button
import pyglet
from pyglet.window import mouse
import activity_recognizer as recognizer

WINDOW_SIZE_X = 1000
WINDOW_SIZE_Y = 800


class FitnessTrainer(pyglet.window.Window):
    def __init__(self):
        # set up the window and the labels
        super().__init__(WINDOW_SIZE_X, WINDOW_SIZE_Y, "Fitness Trainer")

        self.recognizer = recognizer.ActivityRecognizer()
        self.started = False

        self.title_label = pyglet.text.Label(
            "Fitness Trainer",
            x=self.width // 2,
            y=self.height - 50,
            anchor_x="center",
            anchor_y="center",
            font_size=34,
        )
        self.info_label = pyglet.text.Label(
            "Press Start Training, then move the sensor.",
            x=self.width // 2,
            y=self.height - 90,
            anchor_x="center",
            anchor_y="center",
            font_size=14,
        )
        self.status_label = pyglet.text.Label(
            self.recognizer.get_status_text(),
            x=self.width // 2,
            y=200,
            anchor_x="center",
            anchor_y="center",
            font_size=14,
        )
        self.activity_label = pyglet.text.Label(
            self.recognizer.get_activity_text(),
            x=self.width // 2,
            y=150,
            anchor_x="center",
            anchor_y="center",
            font_size=14,
        )
        self.prediction_label = pyglet.text.Label(
            self.recognizer.get_prediction_text(),
            x=self.width // 2,
            y=125,
            anchor_x="center",
            anchor_y="center",
            font_size=18,
        )
        self.correctness_label = pyglet.text.Label(
            self.recognizer.get_correctness_text(),
            x=self.width // 2,
            y=100,
            anchor_x="center",
            anchor_y="center",
            font_size=16,
        )

        button_x = (self.width - 220) // 2
        button_y = (self.height - 70) // 2
        self.button = pyglet.shapes.Rectangle(button_x, button_y, 220, 70, color=(70, 120, 180))
        self.button_label = pyglet.text.Label(
            "Start Training",
            x=self.width // 2,
            y=button_y + 35,
            anchor_x="center",
            anchor_y="center",
            font_size=18,
        )

        # update the screen on a fixed timer
        pyglet.clock.schedule_interval(self.update, 1 / 30.0)

    def update(self, dt):
        # refresh the prediction after training starts
        if self.started:
            self.recognizer.refresh_sensor_values(dt)
            self.status_label.text = self.recognizer.get_status_text()
            self.activity_label.text = self.recognizer.get_activity_text()
            self.prediction_label.text = self.recognizer.get_prediction_text()
            self.correctness_label.text = self.recognizer.get_correctness_text()

    def on_draw(self):
        # draw the full window
        self.clear()
        self.title_label.draw()
        self.info_label.draw()
        self.button.color = (70, 180, 90) if self.started else (70, 120, 180)
        self.button.draw()
        self.button_label.draw()
        self.status_label.draw()
        self.activity_label.draw()
        self.prediction_label.draw()
        self.correctness_label.draw()

    def on_mouse_press(self, x, y, button, modifiers):
        # start training when the button is clicked
        if button != mouse.LEFT or self.started:
            return

        inside_x = self.button.x <= x <= self.button.x + self.button.width
        inside_y = self.button.y <= y <= self.button.y + self.button.height
        if inside_x and inside_y:
            self.started = self.recognizer.start_training()
            self.status_label.text = self.recognizer.get_status_text()

    def on_close(self):
        # close the sensor before leaving
        self.recognizer.close()
        super().on_close()


def main():
        # run the window
    FitnessTrainer()
    pyglet.app.run()


if __name__ == "__main__":
    main()