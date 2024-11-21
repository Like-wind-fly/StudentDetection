# from sympy.codegen import Print
#
# from ultralytics import YOLO
#
# if __name__ == '__main__':
# # Load a pretrained YOLO11n model
#     model = YOLO(model="yolo11n.pt")
#
# # Define path to video file
#     source = r"D:\\code\\dataset\\people\\20sec.mp4"
#
# # Run inference on the source
#     model.predict(source=r"D:\\code\\dataset\\people\\20sec.mp4", stream=True, save_crop=True,save=True)  # generator of Results
#
#     print("results")
#
#

from ultralytics import YOLO
from collections import defaultdict
import cv2

model = YOLO("D:\code\yolo11\stage3\\train\weights\\best.pt")
video_path = "D:\\code\\dataset\\people\\frontleftcorner.avi"

# 打开视频文件
cap = cv2.VideoCapture(video_path)

frame_rate_divider = 10  # 设置帧率除数
frame_count = 0  # 初始化帧计数器

counts = defaultdict(int)
object_str = ""
index = 0

while cap.isOpened():  # 检查视频文件是否成功打开
    ret, frame = cap.read()  # 读取视频文件中的下一帧,ret 是一个布尔值，如果读取帧成功
    if not ret:
        break

    # 每隔 frame_rate_divider 帧进行一次预测
    if frame_count % frame_rate_divider == 0:
        results = model(source=frame, save_crop=True)

        key = f"({index}): "
        index = index + 1
        for result in results:
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                counts[class_id] += 1

        object_str = object_str + ". " + key
        for class_id, count in counts.items():
            object_str = object_str + f"{count} {class_id},"
            counts = defaultdict(int)

    frame_count += 1  # 更新帧计数器

object_str = object_str.strip(',').strip('.')
print("reuslt:", object_str)

cap.release()
cv2.destroyAllWindows()