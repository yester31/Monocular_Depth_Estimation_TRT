import os
import cv2

def extract_frames_from_video(video_path, save_folder, max_count = None):
    # Create output folder if it doesn't exist
    if max_count is not None:
        save_folder = f"{save_folder}_{max_count}"

    os.makedirs(save_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if max_count is None:
        max_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("비디오를 열 수 없습니다:", video_path)
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        # Save frame as PNG
        output_path = os.path.join(save_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(output_path, frame)
        frame_count += 1

        if frame_count > max_count:
            break

    cap.release()
    print(f"{frame_count}개의 프레임을 저장했습니다.")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    save_folder = os.path.join(CUR_DIR, 'video_frames')

    extract_frames_from_video(f"{CUR_DIR}/video/video2.mp4", save_folder)
    # extract_frames_from_video(f"{CUR_DIR}/video/video2.mp4", save_folder, 10)