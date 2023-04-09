import tkinter as tk
import threading
import cv2
import numpy as np
import pyautogui
import time
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from  hubconf import custom
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
import random

def extract_bounding_boxes(predictions):
    bboxes, confidences, classNames = [], [], []
    for item in predictions:
        df = pd.DataFrame(item)
        df.columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']
        # df = df[df['confidence'] > 0.7]  # filter out low confidence detections
        # df = df[df['name'] == 'agility-hotspot']
        for index, detection in df.iterrows():
            xmin, ymin, xmax, ymax = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
            confidences.append(detection['confidence'])
            classNames.append(detection['name'])
    return bboxes, confidences, classNames

def sort_tracks_by_distance(coordsOfCenterPoint, tracks):
    sorted_tracks_dict = {}
    center_point = np.array([coordsOfCenterPoint[0], coordsOfCenterPoint[1]])
    for track in tracks:
        if not track.is_confirmed():
            continue
        # ltrb stands for "left, top, right, bottom", which are the coordinates of a bounding box surrounding an object in an image.
        # track.to_ltrb() is a method of the track object in the DeepSort tracker, 
        # which returns the current bounding box of the object represented by the track in the form of a 
        # tuple containing the (left, top, right, bottom) coordinates. 
        # This method is used to retrieve the current position of the tracked object in the image.
        ltrb = track.to_ltrb()
        # calculate the center point for each bounding box
        bbox_center = np.array([(ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2])
        distance = np.linalg.norm(center_point - bbox_center)
        sorted_tracks_dict[track.track_id] = {"ltrb": ltrb, "bbox_center": bbox_center, "distance": distance}
    # Sort the dictionary by distance and return a list of its items
    sorted_items = sorted(sorted_tracks_dict.items(), key=lambda x: x[1]["distance"])
    return sorted_items

class WindowCapture:
    def __init__(self):
        self.recording = False
        self.clicked_hotspots = [] # so we can avoid clicking the same hotspot over and over again
        self.tracker = DeepSort(max_age=1, max_iou_distance=2.5, n_init=3,
                                nn_budget=10)
        # information on the capture window
        self.top_left_x_window_capture = None
        self.top_left_y_window_capture = None
        self.width_window_capture = None
        self.height_window_capture = None
        self.frame_center = None
        # threads
        self.recording_thread = None
        # time stamps for when xp is detected
        self.xp_detected_log = []

    def move_mouse_and_click(self, data, window_position):
        # Sometimes when traversing the rooftop after using an agility-hotspot the old
        # agility-hotspot might remain the closest object. Which clicking the same
        # spot over and over again is not desirable
        for obj_id, obj_data in data:
            if obj_id in self.clicked_hotspots:
                # Skip over objects that have already been clicked
                print(f"{obj_id} has already been clicked")
                continue

            self.clicked_hotspots.append(obj_id)
            print(f"{obj_id} has been added to the list")
            # Click on the second object and append obj_id to clicked_hotspots list
            bbox_width = obj_data["ltrb"][2] - obj_data["ltrb"][0]
            bbox_height = obj_data["ltrb"][3] - obj_data["ltrb"][1]
            random_x_offset = bbox_width * random.uniform(-.15, .15)
            random_y_offset = bbox_height * random.uniform(-0.15, 0.15)
            random_point = (round(obj_data["ltrb"][0] + bbox_width / 2 + random_x_offset),
                            round(obj_data["ltrb"][1] + bbox_height / 2 + random_y_offset))
            random_point = (random_point[0] + window_position.topleft[0],
                            random_point[1] + window_position.topleft[1])
            original_position = pyautogui.position()
            pyautogui.moveTo(random_point)
            pyautogui.click()
            pyautogui.moveTo(original_position)
            print(f"Current objective {obj_id}")
            print(f"The list of self.clicked_hotspots: {self.clicked_hotspots}")
            break

    def start_recording(self, window_title):
        self.recording = True
        self.recording_thread = threading.Thread(target=self._record_screen, args=(window_title,), daemon=True)
        self.recording_thread.start()

    def stop_recording(self):
        self.recording = False

    def _record_screen(self, window_title):
        window_pos = pyautogui.getWindowsWithTitle(window_title)[0]
        if window_pos is None:
            print("Window not found")
            return
        if window_pos.isMinimized:
            window_pos.restore()
        top_left_x, top_left_y, width, height = window_pos.topleft[0], window_pos.topleft[1], window_pos.width, window_pos.height
        # set our capture window info for calculating and storing center point. 
        self._set_window_variables(top_left_x=top_left_x, top_left_y=top_left_y, width=width, height=height)
        # Create OpenCV window
        cv2.namedWindow("Bot", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bot", 900, 600)
        can_click = True
        last_click_time = time.time()
        # timing variable for periodically clearing out self.clicked_hotspots
        # in case of a misclick situation we may occasionally want to click an object
        # that has already been clicked. misclicks happen irl so it is kind of beneficial that
        # they happen every so often in our application
        last_clear_timer = time.time()
        last_log_clear = time.time()
        xp_detected = False
        xp_detection_delay = .8
        xp_detected_time = None
        # Create background subtractor object with history of 100 frames
        history = 250
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history,10,False)
        while self.recording:
            img = pyautogui.screenshot(region=(top_left_x, top_left_y, width, height))
            # for our background mask there is the second one
            frame = np.array(img)
            frame2 = np.array(img)
            frame3 = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            # Show where hardcoded avatar position is
            cv2.circle(frame, self.frame_center, 5, (255, 255, 0), -1)
            results = model(img, size=640)
            predictions = results.pandas().xyxy

            bboxes, confidences, classNames = extract_bounding_boxes(predictions)
            # shaping the bounding box info for deepSort tracker. we really only need to track
            # agility-hotspot
            formatted_bbs = [([box[0], box[1], box[2] - box[0], box[3] - box[1]], confidence, class_name) 
                            for box, confidence, class_name in zip(bboxes, confidences, classNames) 
                            if class_name == "agility-hotspot" and confidence > 0.6]
            xp_bounding_box = [(box, confidence, class_name)
                            for box, confidence, class_name in zip(bboxes, confidences, classNames)
                            if class_name == "xp" and confidence > 0.3]

            tracks = self.tracker.update_tracks(formatted_bbs, frame=frame)
            tracked_objects_sorted = sort_tracks_by_distance(self.frame_center, tracks)
            
            # display the info in frame
            for i, object in enumerate(tracked_objects_sorted):
                track_id = object[0]
                track_data = object[1]
                ltrb = track_data["ltrb"]
                bbox_center = track_data["bbox_center"]
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID: {track_id}, agility-hotspot", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if i == 0:
                    line_color = (0, 255, 0)
                else:
                    line_color = (0, 0, 255)
                cv2.line(frame, tuple(map(int, self.frame_center)), tuple(map(int, bbox_center)), line_color, 2)
            for object in xp_bounding_box:
                x1, y1, x2, y2 = map(int, tuple(object[0]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 216, 173), 2)
                cv2.putText(frame, f"{object[2]}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 2)
            # Apply background subtraction to get foreground mask
            fg_mask = bg_subtractor.apply(frame2)

            # Count the number of white pixels in the foreground mask
            pixel_count = fg_mask.shape[0] * fg_mask.shape[1]
            white_pixels = cv2.countNonZero(fg_mask)

            # Check if a certain percentage of the pixels have changed in the foreground mask
            if white_pixels / pixel_count > 0.1:  # Change the threshold value as needed
                text = "Running"
                cv2.putText(frame, text, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                can_click = False
            else:
                text = "Stationary"
                cv2.putText(frame, text, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                can_click = True
            
            # we need to get the time that the first xp object is detected and then only after a little delay
            # set the can_click to true. Reason being when the xp first gets detected the screen is still
            # moving too much from the game's movment animations and so by the time we move our
            # cursor to the bounding box the screen has shifted a little bit and we miss our target.
            if len(xp_bounding_box) > 0:
                self.xp_detected_log.append(time.time())
            if len(xp_bounding_box) == 0 and time.time() - last_log_clear > 2:
                self.xp_detected_log.clear()
                last_log_clear = time.time()
            
            if len(self.xp_detected_log) > 0:
                if time.time() - self.xp_detected_log[0] > xp_detection_delay:
                    if time.time() - last_click_time > 3:
                        self.move_mouse_and_click(tracked_objects_sorted, window_pos)
                        last_click_time = time.time()

            if can_click:
                if time.time() - last_click_time > 3:
                    self.move_mouse_and_click(tracked_objects_sorted, window_pos)
                    last_click_time = time.time()

            # clear the list that prevents us from clicking the same spot twice within a time threshold
            # sometimes when traversing an obsticle that same obsticle is still the closest one and you
            # you don't want to keep trying to go backwards.
            if time.time() - last_clear_timer > 25:
                # clearing the list every 25 seconds
                self.clicked_hotspots.clear()
                last_clear_timer = time.time()

            cv2.imshow("Bot",frame)
            # Show the foreground mask
            cv2.imshow("Foreground Mask", fg_mask)

            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                self.recording = False
                break

        cv2.destroyAllWindows()

    def _set_window_variables(self,top_left_x, top_left_y, width, height):
        # tweak this to hard code avatar position when zoomed out
        # currently there seems to be a little difference between pyautogui
        # and the pixel coordinates where as the y isn't quite right
        # needing to go further down the window or in other words needs to be larger
        # value because y coordinates start counting from the top and get larger
        # as they go down the screen
        center_x_percentage = 0.5  # 50% of the width
        center_y_percentage = 0.53  # 53% of the height
        self.frame_center = (int(width * center_x_percentage), int(height * center_y_percentage))
        self.top_left_x_window_capture = top_left_x
        self.top_left_y_window_capture = top_left_y

class App:
    def __init__(self, master):
        self.master = master
        self.master.geometry("400x300")
        self.master.title("bot")
        self.recorder = WindowCapture()
        self.label = tk.Label(master, text="Enter window title")
        self.label.pack(fill=tk.NONE, expand=False, side=tk.TOP, padx=10, pady=10)
        self.window_title_entry = tk.Entry(master)
        self.window_title_entry.insert(0, "moto g fast")
        self.window_title_entry.pack(fill=tk.NONE, expand=False, side=tk.TOP, padx=10, pady=5)
        self.start_button = tk.Button(master, text="Start", command=self.start_recording)
        self.stop_button = tk.Button(master, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.start_button.pack(fill=tk.BOTH, expand=True, side=tk.TOP, padx=10, pady=5)
        self.stop_button.pack(fill=tk.BOTH, expand=True, side=tk.TOP, padx=10, pady=5)
        self.master.protocol("WM_DELETE_WINDOW", self.on_app_close)

    def start_recording(self):
        # Stop any existing recording or clearing of the dict
        self.recorder.stop_recording()

        window_title = self.window_title_entry.get()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.recorder.start_recording(window_title)

    def stop_recording(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.recorder.stop_recording()
        # Clear the clicked hotspots list when the recording is stopped
        self.recorder.clicked_hotspots.clear()

    def on_app_close(self):
        self.recorder.stop_recording()
        self.recorder.recording_thread.join()
        self.master.destroy()

if __name__ == "__main__":
    device = torch.device('cuda')
    if torch.cuda.is_available():
        print("CUDA is available! PyTorch is using the GPU.")
    else:
        print("CUDA is not available. PyTorch is using the CPU.")
    model = custom(path_or_model="runs/train/agility_priff_detector2/weights/best.pt")  # custom example
    root = tk.Tk()
    app = App(root)
    root.mainloop()