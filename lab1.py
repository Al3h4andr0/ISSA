import copy

import cv2
import numpy as np


class Window:
    lane_color = (255, 0, 0)
    lane_width = 3

    def __init__(self, name, frame=None, shape=(1920, 1080), offset=(0, 0)):
        self.name = name
        self.frame = frame
        self.shape = shape
        self.frame = frame
        # Lane points
        self.left_top = self.left_bottom = self.right_top = self.right_bottom = (0, 0)
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

        magic_matrix = cv2.getPerspectiveTransform(np.float32(coordinates),
                                                   np.float32([(width, 0), (0, 0), (0, height), (width, height)]))
        self.frame = cv2.warpPerspective(self.frame, magic_matrix, self.shape)

    def blur(self):
        self.frame = cv2.blur(self.frame, ksize=(5, 5))

    def sobel(self):
        sobel_vertical_kernel = np.float32([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]])
        sobel_horizontal_kernel = np.transpose(sobel_vertical_kernel)

        sobel_vertical_frame = cv2.filter2D(self.frame.astype(np.float32), -1, sobel_vertical_kernel)
        sobel_horizontal_frame = cv2.filter2D(self.frame.astype(np.float32), -1, sobel_horizontal_kernel)

        sobel_frame = np.sqrt(
            (sobel_vertical_frame * sobel_vertical_frame) + (sobel_horizontal_frame * sobel_horizontal_frame))
        self.frame = cv2.convertScaleAbs(sobel_frame)

        return sobel_vertical_frame, sobel_horizontal_frame

    def binarize(self, threshold=int(255 / 2)):
        self.frame = cv2.threshold(self.frame, threshold, 255, cv2.THRESH_BINARY)[1]

    def draw_lanes_treshold(self):
        self.__update_points()
        # left lane
        self.frame = cv2.line(self.frame, self.left_bottom, self.left_top, Window.lane_color, Window.lane_width)
        # right lane
        self.frame = cv2.line(self.frame, self.right_bottom, self.right_top, Window.lane_color, Window.lane_width)

    def __get_white_points(self):
        height, width = self.frame.shape
        num_cols_to_remove = int(width * 0.05)

        new_frame = self.frame.copy()

        new_frame[:, :num_cols_to_remove] = 0
        new_frame[:, -num_cols_to_remove:] = 0

        # Define the width and height of the image
        height, width = self.frame.shape

        # Define the region of interest for each half of the frame
        left_roi = self.frame[:, :int(width / 2)]
        right_roi = self.frame[:, int(width / 2):]

        # Get the x and y coordinates of the white pixels in the left ROI
        left_y, left_x = np.where(left_roi == 255)

        # Shift the x coordinates to account for the left ROI being in the left half of the frame
        # left_x = left_x + int(width / 2)

        # Get the x and y coordinates of the white pixels in the right ROI
        right_y, right_x = np.where(right_roi == 255)
        right_x = right_x + int(width / 2)

        return left_x, left_y, right_x, right_y

    def __update_points(self):
        left_x, left_y, right_x, right_y = self.__get_white_points()
        left_fit, right_fit = Utils.get_linear_regression(left_x, left_y, right_x, right_y)
        height, width = frame.shape[:2]

        y_min = 0
        y_max = height - 1

        x_min_left = int(np.polyval(left_fit, y_min))  # 0 = y_min
        x_max_left = int(np.polyval(left_fit, y_max))

        x_min_right = int(np.polyval(right_fit, 0))  # 0 = y_min
        x_max_right = int(np.polyval(right_fit, height - 1))

        if -1e8 <= x_min_left <= 1e8:
            self.left_bottom = (x_min_left, y_min)
        if -1e8 <= x_max_left <= 1e8:
            self.left_top = (x_max_left, y_max)

        if -1e8 <= x_min_right <= 1e8:
            self.right_bottom = (x_min_right, y_min)
        if -1e8 <= x_max_right <= 1e8:
            self.right_top = (x_max_right, y_max)


def remake_frame(default_frame, black_white_frame, left_bottom, left_top, right_bottom, right_top):
    # matrix with either 0 or 255 - black or white. Will be used as a mask for the image
    lane_mask = np.zeros(default_frame.shape[:2], dtype=np.uint8)
    # left lane
    lane_mask = cv2.line(lane_mask, left_bottom, left_top, 255, Window.lane_width)
    # right lane
    lane_mask = cv2.line(lane_mask, right_bottom, right_top, 255, Window.lane_width)

    height, width = lane_mask.shape[:2]

    # the ones from the trapezoid
    upper_left = (int(width * 0.44), int(height * 0.8))
    upper_right = (int(width * 0.56), int(height * 0.8))
    lower_left = (0, height)
    lower_right = (width, height)

    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_points = np.float32([upper_left, upper_right, lower_left, lower_right])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transform to the image
    warped_img = cv2.warpPerspective(lane_mask, M, (width, height))

    # Threshold the warped image to get a binary mask
    _, mask = cv2.threshold(warped_img, 1, 255, cv2.THRESH_BINARY)

    # Create a new image of the same shape as default_frame
    # output_frame = np.zeros_like(default_frame)

    # Set the pixels that correspond to the non-zero values in the mask to the desired color
    default_frame[mask != 0] = [0, 0, 255]  # Replace with red color (BGR)


    return default_frame
    #
    # cv2.namedWindow("TEST")
    # cv2.imshow("TEST", warped_img)

class Utils:
    @staticmethod
    def get_linear_regression(left_x, left_y, right_x, right_y):
        left_fit = np.polyfit(left_y, left_x, 1)
        right_fit = np.polyfit(right_y, right_x, 1)

        return left_fit, right_fit


def generate_trapezoid(shape):
    trapezoid = np.zeros((shape[1], shape[0]), dtype=np.uint8)
    height = shape[1]
    width = shape[0]

    y = int(height * 0.55)
    upper_left = (int(width * 0.44), int(height * 0.8))
    upper_right = (int(width * 0.56), int(height * 0.8))
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
        "binarized + lanes",
        "result"
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
        (0, 2 * (360 + 1)),
        (480 + 1, 2 * (360 + 1)),
        (2 * (480 + 1), 2 * (360 + 1)),
        (3 * (480 + 1), 2 * (360 + 1)),
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

        window6 = Window(window_names[5], window5.frame, shape, positions[5])
        window6.blur()
        window6.show()

        window7 = Window(window_names[6], window6.frame, shape, positions[6])
        window7.sobel()
        window7.show()

        window8 = Window(window_names[7], window7.frame, shape, positions[7])
        window8.binarize()
        window8.show()

        window9 = Window(window_names[8], window8.frame, shape, positions[8])
        window9.draw_lanes_treshold()
        window9.show()

        lane_frame = remake_frame(window.frame, window9.frame, window9.left_bottom, window9.left_top, window9.right_bottom, window9.right_top)
        window10 = Window(window_names[9], lane_frame, shape, positions[9])
        window10.show()
        # get_white_points(window8.frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
