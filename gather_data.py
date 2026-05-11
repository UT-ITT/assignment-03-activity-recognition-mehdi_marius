# this program gathers sensor data
from DIPPID import SensorUDP
import pandas as pd
from time import sleep
import glob

# Configuration variables - modify these instead of entering them each time
user_name = 'marius'
activity_name = 'jumpingjacks'

# set columns  id, timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
columns = ['id', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# use UPD (via WiFi) for communication
PORT = 5700
sensor = SensorUDP(PORT)

# initilaize empty dataframe
df = pd.DataFrame(columns=columns)
print(df)

# Configuration is set at the top of the script

# wait for the first press to start collecting
print('Waiting for button 1...')
print('button_1: ', sensor.get_value('button_1'))
while not (sensor.get_value('button_1') == 1):
    sleep(0.1)

print('Button 1 pressed, starting data collection')
start_time = pd.Timestamp.now()
while (pd.Timestamp.now() - start_time).total_seconds() < 10:
    print('capabilities: ', sensor.get_capabilities())
    if sensor.has_capability('accelerometer') and sensor.has_capability('gyroscope'):
        # get sensor data
        acc_data = sensor.get_value('accelerometer')
        gyro_data = sensor.get_value('gyroscope')
        timestamp = pd.Timestamp.now()
        # get detailed sensor data for the axis
        acc_x = acc_data['x']
        acc_y = acc_data['y']
        acc_z = acc_data['z']
        gyro_x = gyro_data['x']
        gyro_y = gyro_data['y']
        gyro_z = gyro_data['z']
        # add to the dataframe
        df.loc[len(df)] = {
            'id': len(df) + 1,
            'timestamp': timestamp,
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
        }

        print(df.tail(1))
sensor.disconnect()
# save the data with the according name and number
df['id'] = range(1, len(df) + 1)
# find next number for filename
existing_files = glob.glob(f'data/{user_name}-{activity_name}-*.csv')
next_number = len(existing_files) + 1
# save processed data to CSV file
df.to_csv(f'data/{user_name}-{activity_name}-{next_number}.csv', index=False)
