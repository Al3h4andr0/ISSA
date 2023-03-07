import copy

import cv2
import numpy as np


class Window:
    def __init__(self, name, frame=None, shape=(1920, 1080), offset=(0, 0)):
        self.name = name
        self.frame = frame
        self.shape = shape
        self.frame = frame
        cv2.namedWindow(self.name)
        cv2.moveWindow(self.name, offset[0], offset[1])

    def show(self):
        if self.frame is None:
            raise Exception("Window was shown but the frame is empty")
        cv2.imshow(self.name, self.frame)

    def resize(self):
        self.frame = cv2.resize(self.frame, shape)

    def grayscale(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def stretch(self, coordinates):
        height = self.shape[1]
        width = self.shape[0]

        magic_matrix = cv2.getPerspectiveTransform(np.float32(coordinates), np.float32([(width, 0), (0, 0), (0, height), (width, height)]))
        self.frame = cv2.warpPerspective(self.frame, magic_matrix, shape)

    def blur(self):
        self.frame = cv2.blur(self.frame, ksize=(5, 5))

    def sobel(self):
        sobel_vertical_kernel = np.float32([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]])
        sobel_horizontal_kernel = np.transpose(sobel_vertical_kernel)

        sobel_vertical_frame = cv2.filter2D(self.frame.astype(np.float32), -1, sobel_vertical_kernel)
        sobel_horizontal_frame = cv2.filter2D(self.frame.astype(np.float32), -1, sobel_horizontal_kernel)

        sobel_frame = np.sqrt((sobel_vertical_frame * sobel_vertical_frame) + (sobel_horizontal_frame * sobel_horizontal_frame))
        self.frame = cv2.convertScaleAbs(sobel_frame)

        return sobel_vertical_frame, sobel_horizontal_frame

    def binarize(self, threshold=int(255/2)):
        self.frame = cv2.threshold(self.frame, threshold, 255, cv2.THRESH_BINARY)[1]

def generate_trapezoid(shape):
    trapezoid = np.zeros((shape[1], shape[0]), dtype=np.uint8)
    height = shape[1]
    width = shape[0]

    y = int(height * 0.55)
    upper_left = (int(width * 0.44), int(height*0.8))
    upper_right = (int(width * 0.56), int(height*0.8))
    lower_left = (0, height)
    lower_right = (width, height)

    coordinates = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
    trapezoid = cv2.fillConvexPoly(trapezoid, coordinates, 1)
    return trapezoid, coordinates


if __name__ == "__main__":
    window_names = [
        "original",
        "resized",
        "gray_scale",
        "road",
        "stretched_road",
        "blurred",
        "sobel",
        "binarized",
    ]
    positions = [
        (0, 0),
        (480 + 1, 0),
        (2 * (480 + 1), 0),
        (3 * (480 + 1), 0),
        (0, 360 + 1),
        (480 + 1, 360 + 1),
        (2 * (480 + 1), 360 + 1),
        (3 * (480 + 1), 360 + 1),
        (0, 2 * (360 + 1))
    ]
    shape = (480, 360)

    cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

    while True:
        ret, frame = cam.read()

        window = Window(window_names[0], frame, shape, positions[0])
        window.resize()
        window.show()

        window2 = Window(window_names[1], window.frame, shape, positions[1])
        window2.grayscale()
        window2.show()

        trapezoid, coords = generate_trapezoid(shape)

        window3 = Window(window_names[2], trapezoid * 255, shape, positions[2])
        window3.show()

        window4 = Window(window_names[3], trapezoid * window2.frame, shape, positions[3])
        window4.show()

        window5 = Window(window_names[4], window4.frame, shape, positions[4])
        window5.stretch(coords)
        window5.show()

        cv2.namedWindow("TEST")

        cv2.imshow("TEST", window5.frame)

        window6 = Window(window_names[5], window5.frame, shape, positions[5])
        window6.blur()
        window6.show()

        window7 = Window(window_names[6], window6.frame, shape, positions[6])
        window7.sobel()
        window7.show()

        window8 = Window(window_names[7], window7.frame, shape, positions[7])
        window8.binarize()
        window8.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
