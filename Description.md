# Exercise 03

## `activity_recognizer.py`
- Loads all CSV files from `data/all_csvs`.
- Creates simple features from the sensor columns using mean, standard deviation, minimum, and maximum.
- Trains a `RandomForestClassifier` when the program starts.
- Splits the data into training and test sets and prints the test accuracy.
- Connects to the DIPPID sensor when training starts.
- Stores recent accelerometer and gyroscope readings in a buffer.
- Predicts the current activity from the buffered sensor data.
- Calculates a confidence value and reports whether the model is sure enough.

## `fitness_trainer.py`
- Opens a pyglet window.
- Shows one button: **Start Training**.
- Starts the sensor connection when the button is clicked.
- Refreshes the prediction text on a timer.
- Displays the current activity, prediction, and correctness status.
- Confidence goes up when the activity done is executed right
- Closes the sensor connection when the window is closed.

## Sources
- Pyglet Codeframe from old assignemtns
- Some code refactor and debugging from Copilot
