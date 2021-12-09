import cv2

def get_video(video_path):
    vid_capture = cv2.VideoCapture(video_path)
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    else:
        fps = vid_capture.get(5)
        print('Frames per second : ', fps,'FPS')
        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)
        
def show_video(video_path):
    vid_capture = cv2.VideoCapture(video_path)
    while(vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(20)

            if key == ord('q'):
                break
        else:
            break
    vid_capture.release()
    cv2.destroyAllWindows()