"""model training and dippid sensor handling for the fitness trainer"""

# load the files and build one feature row per csv
from glob import glob
from pathlib import Path

import pandas as pd
from DIPPID import SensorUDP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PORT = 5700
WINDOW_SIZE = 50
CONFIDENCE_THRESHOLD = 0.7
TRAINING_DATA_DIR = Path("data/all_csvs")


def _feature_row(frame):
    # create simple summary values for each sensor column
    features = {}
    for column in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        series = frame[column]
        features[f"{column}_mean"] = series.mean()
        features[f"{column}_std"] = series.std(ddof=0)
        features[f"{column}_min"] = series.min()
        features[f"{column}_max"] = series.max()
    return features


def _load_training_data():
    # read all csv files and use the file name as label
    data_folder = TRAINING_DATA_DIR
    rows = []
    labels = []

    for file_name in sorted(glob(str(Path(data_folder) / "*.csv"))):
        frame = pd.read_csv(file_name)
        rows.append(_feature_row(frame))
        labels.append(Path(file_name).name.split("-")[1])

    return pd.DataFrame(rows), labels


class ActivityRecognizer:
    def __init__(self):
        # prepare sensor state and train the model once
        self.sensor = None
        self.current_acc = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.current_gyro = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.buffer = []
        self.current_prediction = "unknown"
        self.last_confidence = 0.0
        self.target_activity = "running"
        self.target_acivity = self.target_activity
        self.sensor_status = "Model loaded, press Start Training"

        features, labels = _load_training_data()
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(train_x, train_y)
        predictions = self.model.predict(test_x)
        self.accuracy = accuracy_score(test_y, predictions)
        print(f"test accuracy: {self.accuracy:.2f}")

    def start_training(self):
        # connect to dippid when the user starts training
        if self.sensor is None:
            try:
                self.sensor = SensorUDP(PORT)
                self.sensor_status = f"do {self.target_activity}"
            except Exception as exc:
                self.sensor_status = f"DIPPID connection failed: {exc}"
                self.sensor = None
                return False
        return True

    def refresh_sensor_values(self, dt=0.0):
        # read the latest sensor values and keep a short history
        if self.sensor is None:
            return

        if self.sensor.has_capability("accelerometer"):
            value = self.sensor.get_value("accelerometer")
            if isinstance(value, dict):
                self.current_acc = value

        if self.sensor.has_capability("gyroscope"):
            value = self.sensor.get_value("gyroscope")
            if isinstance(value, dict):
                self.current_gyro = value

        self.buffer.append(
            {
                "acc_x": self.current_acc["x"],
                "acc_y": self.current_acc["y"],
                "acc_z": self.current_acc["z"],
                "gyro_x": self.current_gyro["x"],
                "gyro_y": self.current_gyro["y"],
                "gyro_z": self.current_gyro["z"],
            }
        )
        if len(self.buffer) > WINDOW_SIZE:
            self.buffer.pop(0)

        if len(self.buffer) == WINDOW_SIZE:
            self.current_prediction, self.last_confidence = self.predict_current_activity()

    def predict_current_activity(self):
        # predict the current activity from the buffered values
        frame = pd.DataFrame(self.buffer)
        features = pd.DataFrame([_feature_row(frame)])
        prediction = self.model.predict(features)[0]
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        return prediction, confidence

    def get_status_text(self):
        # return the current connection status
        return self.sensor_status

    def get_prediction_text(self):
        # return the latest model prediction
        return f"Prediction: {self.current_prediction}"

    def get_activity_text(self):
        # return the current predicted activity
        return f"Activity: {self.current_prediction}"

    def get_correctness_text(self):
        # return the confidence and training accuracy when the model is sure
        if self.last_confidence >= CONFIDENCE_THRESHOLD:
            return f"Sure training accuracy {self.accuracy:.2f}"
        return f"Not sure confidence {self.last_confidence:.2f}"

    def close(self):
        # close the sensor connection if needed
        if self.sensor is not None:
            try:
                self.sensor.disconnect()
            except Exception:
                pass