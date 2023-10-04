import cv2
import time
import numpy as np
from PendulumPeriodDetector import PendulumPeriodDetector
capture = cv2.VideoCapture("test.mp4")  # For webcam. Replace with 'path_to_video.mp4' for video file
import matplotlib.pyplot as plt

_, frame1 = capture.read()
_, frame2 = capture.read()

# Let the user set the origin by clicking on the video frame
origin = None

def set_origin(event, x, y, flags, param):
    global origin
    if event == cv2.EVENT_LBUTTONDOWN:
        origin = (x, y)


cv2.namedWindow("Set Origin")
cv2.setMouseCallback("Set Origin", set_origin)

# Wait until the user sets the origin
while True:
    cv2.imshow("Set Origin", frame1)
    if cv2.waitKey(1) & 0xFF == ord('q') or origin is not None:
        break

cv2.destroyAllWindows()

# Define bounding box using ROI selector
bbox = cv2.selectROI("Select Object", frame1, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Initialize the tracker
tracker = cv2.TrackerCSRT.create()
tracker.init(frame1, bbox)

# Your existing logic variables
time_start = time.time()
prev_moving = False
angle_list = []
stops_list = []
T_half_list = []
stops_count = 0
T_timestamps = []
T_list = []
labels = []
def calculate_angle(relative_x,relative_y):
    ball_position = (relative_x,relative_y)
    pivot_position = (0,0)

    if not pivot_position or not ball_position:
        print("Couldn't detect one or both of the points")
        return None  # Couldn't detect one or both of the points

    # Calculate the angle using trigonometry
    dx = ball_position[0] - pivot_position[0]
    dy = ball_position[1] - pivot_position[1]
    angle_rad = np.arctan(dx/dy)

    return angle_rad

def calc_T() -> tuple[float, float, float]:
    """
    Calculates T.

    Returns a 3-element tuple. First element is average T, second is minimum T,
    third is maximum T.
    """
    if not T_list:  # Check if the list is empty
        return (0, 0, 0)  # Return default values

    T_avg = np.mean(T_list)
    T_min = np.min(T_list)
    T_max = np.max(T_list)
    return (round(T_avg, 2), round(T_min, 2), round(T_max, 2))


p1 = (int(bbox[0]), int(bbox[1]))
p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
object_center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

relative_x = object_center[0] - origin[0]
relative_y = object_center[1] - origin[1]
initial_angle = calculate_angle(relative_x,relative_y)
print(f"Thxe Initial angle is {initial_angle}")
while capture.isOpened() and frame1 is not None and frame2 is not None:
    success, bbox = tracker.update(frame1)
    if success:
        time_total = round(time.time() - time_start, 2)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        object_center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

        relative_x = object_center[0] - origin[0]
        relative_y = object_center[1] - origin[1]

        # Draw bounding box and origin
        cv2.rectangle(frame1, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame1, origin, 5, (0, 0, 255), -1)
        # X-axis (a line to the right)
        cv2.line(frame1, origin, (origin[0] + 100, origin[1]), (0, 0, 255), 2)
        # Y-axis (a line upwards)
        cv2.line(frame1, origin, (origin[0], origin[1] - 100), (0, 255, 0), 2)
        current_angle = calculate_angle(relative_x, relative_y)
        msg = f"MOVING: Relative Position ({relative_x}, {relative_y})" \
              f"Angle: {current_angle}"
        angle_list.append(current_angle)
        stops_list.append(time_total)

    """color = (0, 255, 0)
    prev_moving = True
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(thresh, None, iterations=2)

    # Determine if there's significant movement
    moving_area = np.sum(motion_mask) / 255  # compute total "white" area
    if moving_area > 10:  # you need to define what SOME_THRESHOLD is, maybe 5000 or so
        msg = f"MOVING: Relative Position ({relative_x}, {relative_y})" \
              f"Angle: {current_angle}"

        color = (0, 255, 0)
        prev_moving = True
    else:
        print("did not find it")
        msg = "NOT MOVING"
        color = (0, 0, 255)
        # Only update the period detector if it's not moving
        stops_list.append(time_total)

        if len(stops_list) > 1:
            T_half = round(time_total - stops_list[-2], 2)
        else:
            T_half = 0

        if T_half >= 0.3:
            prev_moving = False
            stops_count += 1
            T_half_list.append(T)

        if stops_count % 2 == 0:
            T_timestamp = time.time()
            T_timestamps.append(T_timestamp)
            # ignore first period, it's way too short (video's fault)
            if len(T_timestamps) > 3:
                T = round(T_timestamp - T_timestamps[-2], 2)
                T_list.append(T)
                labels.append(f"T: {T}")

    cv2.putText(frame1, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Tracking", frame1)
    """
    t0 = cv2.getTickCount()
    frame1 = frame2
    _, frame2 = capture.read()

    if cv2.waitKey(10) == 27:  # exit if 'esc' key is pressed
        break
#T_avg, T_min, T_max = calc_T()
#print(
    #f"Measured {len(T_list)} full periods. T avg: {T_avg}, T min: {T_min}, T max: {T_max}"
#)
#g_avg, g_min, g_max = calc_g()
#print(f"g avg: {g_avg}, g min: {g_min}, g max: {g_max}")
#for i in range(len(angle_list)):
    #print(f"angle:{angle_list[i]},time:{stops_list[i]}")
angle_dict= {}
for i in range(len(angle_list)):
    angle_dict[angle_list[i]]= stops_list[i]
if len(angle_list) != len(stops_list):
    print("The lists have different lengths. Ensure that each stop time has a corresponding angle.")
    exit()


# estimate the period based on the graph
"""period_estimation = 1.5
time_ini = stops_list[0]
splits = []
current_split = {}
climax = {}
for angle,time_stop in angle_dict.items():
    if time_stop-time_ini>period_estimation:
        if current_split:
            splits.append(current_split)
            current_split = {}
            time_ini += period_estimation
    current_split[angle] = time_stop
    splits.append(current_split)
for period in splits:
    max_angle = max([abs(angle) for angle in period.keys()])
    climax[max_angle] = period[max_angle]
climax = dict(sorted(climax.items(), key=lambda item: item[1]))
print(climax)"""




# Create a new figure
plt.figure()

# Plot the data
plt.plot(stops_list, angle_list, 'o-', label='Angle vs. Time')
plt.xlabel('Stop Time (s)')
plt.ylabel('Angle (radians)')
plt.title('Angle vs. Stop Time')
plt.legend()

# Display the plot
plt.show()
capture.release()
cv2.destroyAllWindows()
