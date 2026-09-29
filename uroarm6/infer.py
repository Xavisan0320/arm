import queue
from multiprocessing import Process, Queue, freeze_support

import cv2
import numpy as np
from ultralytics import YOLO


def infer_img(img_que: Queue, result_que: Queue):

    print("Loading YOLO model...", flush=True)

    model = YOLO("contest.pt")

    print("MODEL:", model.names, flush=True)
    print("YOLO model loaded!", flush=True)

    while True:

        bmp, color, shape = img_que.get()

        if bmp is None:
            result_que.put((np.nan, np.nan, None, []))
            break

        # print(
        #     f"image received! target_class={target_class}",
        #     flush=True
        # )

        cv2.imwrite("debug_yolo.jpg", bmp)
        
        results = model.predict(
            source=bmp,
            conf=0.1,
            iou=0.45,
            imgsz=640,
            max_det=20,
            verbose=False
        )

        center_x = np.nan
        center_y = np.nan
        class_name = None

        # Every detection in the frame, whatever its class - the caller needs
        # the other items' positions too, not just the one it asked for (see
        # arm.py's 自動縦掴み: how close the neighbours are decides how the
        # gripper goes in).
        items = []

        best_conf = 0.0

        if len(results) > 0:

            boxes = results[0].boxes

            # print(
            #     f"detected boxes = {len(boxes)}",
            #     flush=True
            # )

            for box in boxes:

                class_id = int(box.cls[0])
                name = model.names[class_id]
                confidence = float(box.conf[0])

                # print(
                #     f"DETECTED: {name} "
                #     f"conf={confidence:.3f} "
                #     f"target={target_class}",
                #     flush=True
                # )

                x1, y1, x2, y2 = \
                    box.xyxy[0].cpu().numpy()

                items.append(
                    (
                        float((x1 + x2) / 2),
                        float((y1 + y2) / 2),
                        name
                    )
                )

                # 色・形状が指定されている場合。クラス名は "{色}_{形状}" 形式
                # ("red_cube", "red_ragby", ...) を前提に、どちらか片方だけの
                # 指定でも絞り込めるようにする。
                if color is not None or shape is not None:

                    name_color, _, name_shape = name.partition('_')

                    if color is not None and name_color != color:
                        continue

                    if shape is not None and name_shape != shape:
                        continue

                # confidenceが一番高いものを選ぶ
                if confidence > best_conf:

                    best_conf = confidence

                    center_x = int(
                        (x1 + x2) / 2
                    )

                    center_y = int(
                        (y1 + y2) / 2
                    )

                    class_name = name

            # if class_name is not None:

                # print(
                #     f"RESULT: {class_name} "
                #     f"x={center_x} "
                #     f"y={center_y} "
                #     f"conf={best_conf:.3f}",
                #     flush=True
                # )

            # else:

                # print(
                #     "RESULT: no target",
                #     flush=True
                # )

        result_que.put(
            (
                center_x,
                center_y,
                class_name,
                items
            )
        )

class Inference:

    def __init__(self):

        self.img_que = Queue()
        self.result_que = Queue()

        self.cx = np.nan
        self.cy = np.nan
        self.class_name = None
        self.items = []          # every detection of the last frame - see all_items()

        self.pw = Process(
            target=infer_img,
            args=(
                self.img_que,
                self.result_que
            )
        )

        self.pw.start()

    def get(self, frame, color=None, shape=None):

        h, w = frame.shape[:2]

        assert h == w

        # --------------------------------
        # 推論画像を送る
        # --------------------------------

        if self.img_que.empty():

            if h == 720:

                frame2 = frame

            else:

                frame2 = cv2.resize(
                    frame,
                    (720, 720)
                )

            self.img_que.put(
                (
                    frame2,
                    color,
                    shape
                )
            )

        # --------------------------------
        # 推論結果を取得
        # --------------------------------

        try:

            cx, cy, class_name, items = \
                self.result_que.get_nowait()

            # 720x720 → 元画像サイズ (the target below is scaled the same way)
            self.items = [
                (ix * w / 720, iy * h / 720, name)
                for ix, iy, name in items
            ]

            # print(
            #     f"GET RESULT: "
            #     f"x={cx}, "
            #     f"y={cy}, "
            #     f"class={class_name}",
            #     flush=True
            # )

            if not np.isnan(cx):

                # 720x720 → 元画像サイズ
                cx = cx * w / 720
                cy = cy * h / 720

                self.cx = cx
                self.cy = cy
                self.class_name = class_name

            else:

                # 今回検出できなかった場合
                self.cx = np.nan
                self.cy = np.nan
                self.class_name = None

        except queue.Empty:

            pass

        return (
            self.cx,
            self.cy,
            self.class_name
        )

    def all_items(self):
        """[(screen x, screen y, class name), ...] for every item detected in
        the last frame, unfiltered by色/形 - get() returns only the one that
        matches, but a caller that has to know how crowded the table is needs
        all of them (arm.py's 自動縦掴み)."""

        return list(self.items)

    def close(self):

        self.img_que.put(
            (None, None, None)
        )

        while True:

            cx, cy, class_name, _ = \
                self.result_que.get()

            if np.isnan(cx):

                break


if __name__ == '__main__':

    freeze_support()

    from util import read_params
    from camera import (
        initCamera,
        getCameraFrame,
        closeCamera
    )

    params = read_params()

    initCamera(params)

    cv2.namedWindow('window')

    inference = Inference()

    # --------------------------------
    # テスト対象
    #
    # None       = 全クラス
    # blue_cube  = 青
    # green_cube = 緑
    # red_cube   = 赤キューブ
    # red_ragby  = 赤ラグビー
    # --------------------------------

    target_color = None
    target_shape = None

    while True:

        frame = getCameraFrame()
        if frame is None:
            continue

        cx, cy, class_name = inference.get(
            frame,
            target_color,
            target_shape
        )

        # --------------------------------
        # 検出結果表示
        # --------------------------------

        if not np.isnan(cx):

            cv2.circle(
                frame,
                (
                    int(cx),
                    int(cy)
                ),
                10,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"{class_name} "
                f"({int(cx)}, {int(cy)})",
                (
                    int(cx) + 15,
                    int(cy)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # print(
            #     f"FOUND: "
            #     f"{class_name} "
            #     f"x={int(cx)} "
            #     f"y={int(cy)}"
            # )

        # --------------------------------
        # 表示
        # --------------------------------

        h, w = frame.shape[:2]

        if h <= 720:

            frame2 = frame

        else:

            frame2 = cv2.resize(
                frame,
                (720, 720)
            )

        cv2.imshow(
            'window',
            frame2
        )

        k = cv2.waitKey(1)

        if k == ord('q'):

            inference.close()

            closeCamera()

            break