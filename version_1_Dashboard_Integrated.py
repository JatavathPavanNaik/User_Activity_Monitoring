import streamlit as st
import cv2
import os
import pandas as pd
from ultralytics import YOLO

class RemoveDuplicatedFrames:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path,task="detect")

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_areas = []
        frame_number = 0

        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            results = self.model.predict(frame)
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            names = results[0].names
            confidences = results[0].boxes.conf.tolist()
            areas = []

            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                class_name = names[int(cls)]
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)

            all_areas.append((frame_number, areas))
            progress_percent = int((frame_number / frame_count) * 100)
            progress_bar.progress(progress_percent)

        cap.release()
        progress_bar.empty()  # Clear progress bar
        st.success("Preprocessing Is Done.......Please wait.......Saving the Video File")
        return all_areas

    @staticmethod
    def find_duplicate_frames(df):
        desired_rows = []

        for index, row in df.iterrows():
            if index > 0:
                prev_row = df.iloc[index - 1]
                if row['No_of_Bounding_Boxes'] == prev_row['No_of_Bounding_Boxes']:
                    areas_current = row['Individual_Areas']
                    areas_prev = prev_row['Individual_Areas']
                    all_diffs_less_than_100 = all(abs(area_curr - area_prev) < 100 for area_curr, area_prev in zip(areas_current, areas_prev))
                    if all_diffs_less_than_100:
                        desired_rows.append(row)

        if desired_rows:
            return pd.DataFrame(desired_rows)
        else:
            st.write("No Duplicated Frames Found")

    @staticmethod
    def delete_frames(input_video_path, output_video_path, frames_to_delete):
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number not in frames_to_delete:
                out.write(frame)
            frame_number += 1

        cap.release()
        out.release()

    @staticmethod
    def extract_frames(video_path, frame_numbers, output_folder):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_number in frame_numbers:
                frames.append(frame)
                cv2.imwrite(f"{output_folder}/frame_{frame_number}.jpg", frame)
            if len(frames) == len(frame_numbers):
                break

        cap.release()
        return frames

    def remove_duplicate_frames(self, video_path, output_video_path, output_frame_folder):
        all_areas = self.process_video(video_path)

        sum_data = [(tup[0], sum(tup[1]), len(tup[1]), tup[1]) for tup in all_areas]
        df = pd.DataFrame(sum_data, columns=['Frame', 'Total_Area', 'No_of_Bounding_Boxes', 'Individual_Areas'])

        duplicate_frames_df = self.find_duplicate_frames(df)
        if duplicate_frames_df is not None:
            frames_to_delete = duplicate_frames_df["Frame"].astype(int).to_list()
            self.delete_frames(video_path, output_video_path, frames_to_delete)
            extracted_frames = self.extract_frames(video_path, frames_to_delete, output_frame_folder)
            st.success("Duplicate frames found and removed.")
            st.write(f"Output video saved at: {output_video_path}")
        else:
            st.write("No duplicate frames found.")

def main():
    st.title("Remove Duplicated Frames")

    yolo_model_path = st.text_input("Enter the path to YOLO model weights:")
    video_path = st.text_input("Enter the path to the input video file:")
    output_video_path = st.text_input("Enter the path to save the output video file:")
    output_frame_folder = st.text_input("Enter the path to save the duplicated frames:")

    output_video_path = os.path.join(output_video_path, "Preprocessed_" + os.path.basename(video_path))

    remover = RemoveDuplicatedFrames(yolo_model_path)

    if st.button("Remove Duplicated Frames"):
        with st.spinner("Processing..."):
            remover.remove_duplicate_frames(video_path, output_video_path, output_frame_folder)

if __name__ == "__main__":
    main()