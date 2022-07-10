import easyocr
import cv2
from easyocr.easyocr import Reader
from termcolor import colored
import numpy as np
import srt
# import subs_fixer as sf


def time_to_hhmmssmmm(seconds_duration, input_type="miliseconds"):
    '''
    Convert a time number into the string format HH:MM:SS:MMM
    Input types: miliseconds (default), seconds, minutes, hours
    '''
    switch_divider = {
        "miliseconds": 3600000,
        "seconds": 3600,
        "minutes": 60,
        "hours": 1
    }
    hours = seconds_duration/switch_divider[input_type]
    hh = int(hours) # Return value
    minutes = (hours-hh)*60
    mm = int(minutes) # Return value
    seconds = (minutes-mm)*60
    ss = int(seconds) # Return value
    miliseconds = (seconds-ss)*1000
    mmm = int(miliseconds) # Return value
    return str("%02d" % hh) + ":" + str("%02d" % mm) + ":" + str("%02d" % ss) + "," + str("%03d" % mmm)


def get_frame_position(cv_frame, position="bottom"):
    '''
    Get a horizontal frinje of an opencv frame.
    position:
        bottom (default): lower quarter
        top: upper quarter
    '''
    height, _, _ = cv_frame.shape
    frame_position = ""
    if position == "bottom":
        frame_position = cv_frame[int((height/4)*3):,:]
    if position == "top":
        frame_position = cv_frame[:int((height/4)),:]
    return frame_position


def frame_processing(cv_frame):
    '''
    Improve the frame quality through a channels average.
    '''
    image = np.float32(cv_frame)/255 # Floating point convertion
    height, width, _ = image.shape # Image dimensions

    # Channels generation
    image_bgr = image
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_gs = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Channels average
    channels_list = [image_bgr[:,:,0], image_bgr[:,:,1], image_bgr[:,:,2], image_hsv[:,:,1], image_hsv[:,:,2], image_gs]
    image_avg = np.zeros((height, width), dtype=np.float32)
    for channel in channels_list:
        image_avg += channel
    image_avg /= len(channels_list)

    image_avg = np.uint8(image_avg*255) # Integer convertion
    image_avg = np.expand_dims(image_avg, axis=2) # Channel adding

    # cv2.imshow("Fotograma", image_avg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_avg


def video_processing(video_path):
    ocr_reader = easyocr.Reader(["en", "es"]) # EasyOCR reader: English and Español

    subtitles_top_path = video_path.split(".")[0] + "_top.srt"
    subtitles_bot_path = video_path.split(".")[0] + "_bot.srt"
    with open(subtitles_top_path, "w", encoding="utf8") as f:
        f.write("")
    with open(subtitles_bot_path, "w", encoding="utf8") as f:
        f.write("")

    print(colored("File:", "green"), video_path)
    video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration_seconds = int(frame_count/fps)
    print(colored("Frames per second:", "green"), fps)
    print(colored("Total duration:", "green"), time_to_hhmmssmmm(total_duration_seconds, input_type="seconds"))

    previous_ret, previous_frame = video.read()
    if not previous_ret:
        print(colored("Fin del procesamiento.", "red"))
        return
        
    previous_frame_top_processed = frame_processing(get_frame_position(previous_frame, position="top"))
    previous_frame_bot_processed = frame_processing(get_frame_position(previous_frame, position="bottom"))
    previous_top_txt = " ".join(ocr_reader.readtext(previous_frame_top_processed, detail = 0))
    previous_bot_txt = " ".join(ocr_reader.readtext(previous_frame_bot_processed, detail = 0))

    srt_count_top = 0
    frames_accum_top = np.zeros(previous_frame_top_processed.shape, dtype=np.float32)
    similarity_count_top = 0
    srt_count_bot = 0
    frames_accum_bot = np.zeros(previous_frame_bot_processed.shape, dtype=np.float32)
    similarity_count_bot = 0

    # Subtitles in the first frame top
    if previous_top_txt != "":
        srt_count_top += 1
        with open(subtitles_top_path, "a", encoding="utf8") as f:
            f.write(str(srt_count_top) + "\n" + "00:00:00,000" + " --> ")
        frames_accum_top += np.float32(previous_frame_top_processed)/255
        similarity_count_top += 1

    # Subtitles in the first frame bot
    if previous_bot_txt != "":
        srt_count_bot += 1
        with open(subtitles_bot_path, "a", encoding="utf8") as f:
            f.write(str(srt_count_bot) + "\n" + "00:00:00,000" + " --> ")
        frames_accum_bot += np.float32(previous_frame_bot_processed)/255
        similarity_count_bot += 1

    for position in range(1, total_duration_seconds*fps, 1):
        time = video.get(cv2.CAP_PROP_POS_MSEC) # time in miliseconds
        time = time_to_hhmmssmmm(time)
        print(colored("Frame:", "green"), f"{position}/{total_duration_seconds*fps}", colored("Time:", "green"), time)
        video.set(cv2.CAP_PROP_POS_FRAMES, position)

        ret, frame = video.read()
        if not ret:
            print(colored("Fin del procesamiento.", "red"))
            break
        frame_top_processed = frame_processing(get_frame_position(frame, position="top"))
        frame_bot_processed = frame_processing(get_frame_position(frame, position="bottom"))
        frame_top_txt = " ".join(ocr_reader.readtext(frame_top_processed, detail = 0))
        frame_bot_txt = " ".join(ocr_reader.readtext(frame_bot_processed, detail = 0))
        print(colored("Top subtitle:", "blue"), frame_top_txt)
        print(colored("Bot subtitle:", "blue"), frame_bot_txt)

        # Top
        if frame_top_txt != "": # If there are subtitles
            if previous_top_txt == "": # If there are subtitles but before there were not
                srt_count_top += 1
                with open(subtitles_top_path, "a", encoding="utf8") as f:
                    f.write(str(srt_count_top) + "\n" + time + " --> ")
            else: # If there are subtitles and before there were too
                similarity = cv2.matchTemplate(previous_frame_top_processed, frame_top_processed, cv2.TM_CCORR_NORMED)[0][0]
                if similarity < 0.9: # If there are subtitles and before there were too but they are different
                    print(colored("Similarity TOP:", "red"), similarity)
                    frame_avg = frames_accum_top / similarity_count_top
                    frame_avg = np.uint8(frame_avg*255)
                    srt_text = " ".join(ocr_reader.readtext(frame_avg, detail = 0))
                    print("Subtítulo agregado:", colored(srt_text, "yellow"))
                    srt_count_top += 1
                    with open(subtitles_top_path, "a", encoding="utf8") as f:
                        f.write(time + "\n" + srt_text + "\n\n" + str(srt_count_top) + "\n" + time + " --> ")
                    frames_accum_top = np.zeros(previous_frame_top_processed.shape, dtype=np.float32)
                    similarity_count_top = 0
                else: # If there are subtitles and before there were too and they are the same
                    print(colored("Similarity TOP:", "green"), similarity)
                    frames_accum_top += np.float32(frame_top_processed)/255
                    similarity_count_top += 1
        elif previous_top_txt != "": # If there aren't subtitles but before there were
            frame_avg = frames_accum_top / similarity_count_top
            frame_avg = np.uint8(frame_avg*255)
            srt_text = " ".join(ocr_reader.readtext(frame_avg, detail = 0))
            print("Subtítulo TOP agregado:", colored(srt_text, "yellow"))
            with open(subtitles_top_path, "a", encoding="utf8") as f:
                f.write(time + "\n" + srt_text + "\n\n")
            frames_accum_top = np.zeros(previous_frame_top_processed.shape, dtype=np.float32)
            similarity_count_top = 0

        previous_top_txt = frame_top_txt
        previous_frame_top_processed = frame_top_processed

        # Bottom
        if frame_bot_txt != "": # If there are subtitles
            if previous_bot_txt == "": # If there are subtitles but before there were not
                srt_count_bot += 1
                with open(subtitles_bot_path, "a", encoding="utf8") as f:
                    f.write(str(srt_count_bot) + "\n" + time + " --> ")
            else: # If there are subtitles and before there were too
                similarity = cv2.matchTemplate(previous_frame_bot_processed, frame_bot_processed, cv2.TM_CCORR_NORMED)[0][0]
                if similarity < 0.9: # If there are subtitles and before there were too but they are different
                    print(colored("Similarity BOT:", "red"), similarity)
                    frame_avg = frames_accum_bot / similarity_count_bot
                    frame_avg = np.uint8(frame_avg*255)
                    srt_text = " ".join(ocr_reader.readtext(frame_avg, detail = 0))
                    print("Subtítulo agregado:", colored(srt_text, "yellow"))
                    srt_count_bot += 1
                    with open(subtitles_bot_path, "a", encoding="utf8") as f:
                        f.write(time + "\n" + srt_text + "\n\n" + str(srt_count_bot) + "\n" + time + " --> ")
                    frames_accum_bot = np.zeros(previous_frame_bot_processed.shape, dtype=np.float32)
                    similarity_count = 0
                else: # If there are subtitles and before there were too and they are the same
                    print(colored("Similarity BOT:", "green"), similarity)
                    frames_accum_bot += np.float32(frame_bot_processed)/255
                    similarity_count_bot += 1
        elif previous_bot_txt != "": # If there aren't subtitles but before there were
            frame_avg = frames_accum_bot / similarity_count_bot
            frame_avg = np.uint8(frame_avg*255)
            srt_text = " ".join(ocr_reader.readtext(frame_avg, detail = 0))
            print("Subtítulo BOT agregado:", colored(srt_text, "yellow"))
            with open(subtitles_bot_path, "a", encoding="utf8") as f:
                f.write(time + "\n" + srt_text + "\n\n")
            frames_accum_bot = np.zeros(previous_frame_bot_processed.shape, dtype=np.float32)
            similarity_count_bot = 0

        previous_bot_txt = frame_bot_txt
        previous_frame_bot_processed = frame_bot_processed

    # Subtitles in the last frame top
    if previous_top_txt != "":
        frame_avg = frames_accum_top / similarity_count_top
        frame_avg = np.uint8(frame_avg*255)
        srt_text = " ".join(ocr_reader.readtext(frame_avg, detail = 0))
        print("Subtítulo TOP agregado:", colored(srt_text, "yellow"))
        with open(subtitles_top_path, "a", encoding="utf8") as f:
            f.write(time + "\n" + srt_text + "\n\n")

    # Subtitles in the last frame bot
    if previous_bot_txt != "":
        frame_avg = frames_accum_bot / similarity_count_bot
        frame_avg = np.uint8(frame_avg*255)
        srt_text = " ".join(ocr_reader.readtext(frame_avg, detail = 0))
        print("Subtítulo BOT agregado:", colored(srt_text, "yellow"))
        with open(subtitles_bot_path, "a", encoding="utf8") as f:
            f.write(time + "\n" + srt_text + "\n\n")


def function_batch(videos_folder):
    return

video_path = "video.avi"
video_processing(video_path)

# image_path = "image.png"
# image_processing(image_path)