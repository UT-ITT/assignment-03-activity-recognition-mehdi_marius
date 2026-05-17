"""model training and dippid sensor handling for the fitness trainer"""

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from DIPPID import SensorUDP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PORT = 5700
WINDOW_SIZE = 50
CONFIDENCE_THRESHOLD = 0.7
TRAINING_DATA_DIR = Path("data/all_csvs")


def _preprocess_frame(frame):
    # apply a simple moving average filter to reduce noise
    return frame.rolling(window=5, min_periods=1).mean()


def _feature_row(frame):
    features = {}
    
    acc_mag = np.sqrt(frame["acc_x"]**2 + frame["acc_y"]**2 + frame["acc_z"]**2)
    gyro_mag = np.sqrt(frame["gyro_x"]**2 + frame["gyro_y"]**2 + frame["gyro_z"]**2)
    
    data_columns = {
        "acc_x": frame["acc_x"].values,
        "acc_y": frame["acc_y"].values,
        "acc_z": frame["acc_z"].values,
        "gyro_x": frame["gyro_x"].values,
        "gyro_y": frame["gyro_y"].values,
        "gyro_z": frame["gyro_z"].values,
        "acc_mag": acc_mag.values,
        "gyro_mag": gyro_mag.values
    }

    for name, series in data_columns.items():
        # Time domain features
        features[f"{name}_mean"] = np.mean(series)
        features[f"{name}_std"] = np.std(series)
        features[f"{name}_min"] = np.min(series)
        features[f"{name}_max"] = np.max(series)
        features[f"{name}_var"] = np.var(series)
        features[f"{name}_median"] = np.median(series)
        features[f"{name}_ptp"] = np.ptp(series)
        
        # Frequency domain features
        fft_vals = np.abs(np.fft.rfft(series))
        features[f"{name}_fft_mean"] = np.mean(fft_vals)
        features[f"{name}_fft_std"] = np.std(fft_vals)
        features[f"{name}_energy"] = np.sum(series**2) / len(series)
        features[f"{name}_dom_freq"] = np.argmax(fft_vals[1:]) + 1 if len(fft_vals) > 1 else 0

    # Cross-axis correlations
    with np.errstate(divide='ignore', invalid='ignore'):
        features["acc_xy_corr"] = np.corrcoef(data_columns["acc_x"], data_columns["acc_y"])[0, 1]
        features["acc_xz_corr"] = np.corrcoef(data_columns["acc_x"], data_columns["acc_z"])[0, 1]
        features["acc_yz_corr"] = np.corrcoef(data_columns["acc_y"], data_columns["acc_z"])[0, 1]
        features["gyro_xy_corr"] = np.corrcoef(data_columns["gyro_x"], data_columns["gyro_y"])[0, 1]
        features["gyro_xz_corr"] = np.corrcoef(data_columns["gyro_x"], data_columns["gyro_z"])[0, 1]
        features["gyro_yz_corr"] = np.corrcoef(data_columns["gyro_y"], data_columns["gyro_z"])[0, 1]

    # Handle NaN values resulting from division by zero in corrcoef
    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0

    return features


def _load_training_data():
    data_folder = TRAINING_DATA_DIR
    rows = []
    labels = []
    step_size = 10

    for file_name in sorted(glob(str(Path(data_folder) / "*.csv"))):
        frame = pd.read_csv(file_name)
        
        # Drop the first and last 15% of the data to remove setup/stop noise
        drop_count = int(len(frame) * 0.15)
        if len(frame) > drop_count * 2 + WINDOW_SIZE:
            frame = frame.iloc[drop_count:-drop_count].reset_index(drop=True)
            
        frame = _preprocess_frame(frame)
        
        # Create sliding windows to extract features
        for start in range(0, len(frame) - WINDOW_SIZE + 1, step_size):
            window = frame.iloc[start:start + WINDOW_SIZE]
            rows.append(_feature_row(window))
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
        
        self.activities = ["jumpingjacks", "lifting", "rowing", "running"]
        self.current_activity_idx = 0
        self.target_activity = self.activities[self.current_activity_idx]
        self.correct_time = 0.0
        self.target_time = 5.0  # seconds required to complete the activity
        
        self.sensor_status = "Model loaded, press Start Training"

        features, labels = _load_training_data()
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        train_x, test_x, train_y, test_y = train_test_split(
            features_scaled,
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
            
            if self.current_prediction == self.target_activity and self.last_confidence >= CONFIDENCE_THRESHOLD:
                self.correct_time += dt
            
            if self.correct_time >= self.target_time:
                self.correct_time = 0.0
                self.current_activity_idx = (self.current_activity_idx + 1) % len(self.activities)
                self.target_activity = self.activities[self.current_activity_idx]
                self.sensor_status = f"do {self.target_activity}"

    def predict_current_activity(self):
        # predict the current activity from the buffered values
        frame = pd.DataFrame(self.buffer)
        frame = _preprocess_frame(frame)
        features = pd.DataFrame([_feature_row(frame)])
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = float(max(probabilities))
        return prediction, confidence

    def get_status_text(self):
        # return the current connection status
        return self.sensor_status

    def get_prediction_text(self):
        # return the latest model prediction
        return f"Prediction: {self.current_prediction}"

    def get_activity_text(self):
        # return the current targeted activity
        return f"Target Activity: {self.target_activity}"

    def get_correctness_text(self):
        # return the execution correctness
        progress = min(100, int((self.correct_time / self.target_time) * 100))
        is_correct = "Yes" if (self.current_prediction == self.target_activity and self.last_confidence >= CONFIDENCE_THRESHOLD) else "No"
        return f"Executed Correctly: {is_correct} (Progress: {progress}%)"

    def close(self):
        # close the sensor connection if needed
        if self.sensor is not None:
            try:
                self.sensor.disconnect()
            except Exception:
                pass