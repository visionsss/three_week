import cv2
from skimage import io

if __name__ == '__main__':
    # origin = io.imread('2.png')
    # origin2 = cv2.imread('2.png', cv2.COLOR_BGR2RGB)

    cap = cv2.VideoCapture(0)
    ref = True
    while ref:
        ref, frame = cap.read()
        if ref:
            cv2.imshow("", frame)
            if cv2.waitKey(10) == ord('q'):
                break
