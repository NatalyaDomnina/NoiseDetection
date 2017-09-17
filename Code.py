import cv2
import random
import numpy as np

# create video from the picture
def createVideo(input_img, pathVideo):
    height, width, layers =  input_img.shape
    fours = cv2.VideoWriter_fourcc(*'MJPG')
    frames = 15
    video = cv2.VideoWriter(pathVideo,fours,frames,(width,height))
    for i in range(100):
        video.write(input_img)
    cv2.destroyAllWindows()
    video.release()

# create video with noise from the picture
def createVideoWithNoise(input_img, noise, pathVideo):
    height, width, layers = input_img.shape
    fours = cv2.VideoWriter_fourcc(*'MJPG')
    frames = 15
    video = cv2.VideoWriter(pathVideo, fours, frames, (width, height))

    random.seed(version = 2)
    r = random.randrange(0, 100, 1)
    x, y, c = noise.shape

    for i in range(100):
        if i == r:
            noise_img = input_img.copy()
            ofs_y1 = random.randrange(0, height - x, 1)
            ofs_x1 = random.randrange(0, width - y, 1)
            y1, y2 = ofs_y1, ofs_y1 + noise.shape[0]
            x1, x2 = ofs_x1, ofs_x1 + noise.shape[1]
            alpha_s = noise[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                noise_img[y1:y2, x1:x2, c] = (alpha_s * noise[:, :, c] +
                                              alpha_l * noise_img[y1:y2, x1:x2, c])
            video.write(noise_img)
        else:
            video.write(input_img)

# returns the frame number with an noise
def noiseDetection(pathVideo):
    cap = cv2.VideoCapture(pathVideo)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(video_length):
        _, frame = cap.read()
        num_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame2 = cv2.pyrDown(frame) # !
        frame3 = cv2.pyrDown(frame2)
        frame3 = frame3^2
        frame1_2 = cv2.pyrUp(frame)
        frame2_2 = cv2.pyrUp(frame2)
        frame3_2 = cv2.pyrUp(frame3) # !

        frame2_3_2 = frame2 - frame3_2 # !

        gradX = cv2.Sobel(frame_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(frame_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.blur(gradient, (5, 5))
        (_, thresh) = cv2.threshold(blurred, 215, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        closed = 255 - closed

        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)

        box = np.array(box).reshape((-1, 1, 2)).astype(np.int32)

        if (box[0] - box[1])[0][0] > 20 or (box[0] - box[1])[0][1] > 20:
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)
            res = num_frame
            frame_noise = frame.copy()

        cv2.imshow("Image", frame)
        #cv2.waitKey(0)
        cv2.waitKey(20)
    return res, frame_noise
'''
# путь к текстуре
input_img = cv2.imread('C:/Users/Natali/Documents/132.png')
# путь к помехе
noise = cv2.imread('C:/Users/Natali/Documents/555.png', cv2.IMREAD_UNCHANGED)
# путь к видео (где будет создано)
pathVideo = "C:/Users/Natali/Documents/video.avi"

# создаем чистое видео
createVideo(input_img, pathVideo)
# создаем видео с помехой
createVideoWithNoise(input_img, noise, pathVideo)
# выводим номер кадра с найденной помехой
res = noiseDetection(pathVideo)
num_fr = res[0]
print(num_fr)
# и сам кадр
fr = res[1]
cv2.imshow('000', fr)
cv2.waitKey(0)
'''