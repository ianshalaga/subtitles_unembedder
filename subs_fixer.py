from cv2 import meanStdDev
import srt
from fuzzywuzzy import fuzz
from textblob import TextBlob
import statistics as st
from termcolor import colored

def subtitles_joiner(subs_top_path, subs_bot_path):
    '''
    Join subtitles of top and bottom.
    '''
    # Load Subtitles
    srt_top = ""
    with open(subs_top_path, "r", encoding="utf8") as f:
        srt_top = f.read()
    srt_bot = ""
    with open(subs_bot_path, "r", encoding="utf8") as f:
        srt_bot = f.read()
    
    # Merge subtitles
    srt_top_list = list(srt.parse(srt_top))
    srt_bot_list = list(srt.parse(srt_bot))
    srt_list = srt_top_list + srt_bot_list
    output = srt.compose(srt_list)

    # Save subtitle
    output_path = "_".join(subs_top_path.split(".")[0].split("_")[:-3]) + "joined.srt"
    with open(output_path, "w", encoding="utf8") as f:
        f.write(output)


def characters_per_second(time_start, time_end, text):
    '''
    Calculate characters per second (cps) of a subtitle.
    The time parameters have format: HH:MM:SS.MMMMMM
    '''
    time_delta = (time_end-time_start).total_seconds() # srt library method
    characters = len(text) - text.count(" ")
    cps = int(characters/time_delta)
    return cps


def subtitles_fixer(subs_path):
    srt_bot = ""
    with open(subs_path, "r", encoding="utf8") as f:
        srt_bot = f.read()

    srt_list = list(srt.parse(srt_bot))

    # Delete empties subtitles
    srt_no_empties_list = list()
    for e in srt_list:
        if e.content != "":
            srt_no_empties_list.append(e)

    sub_idx = ""
    sub_start = ""
    sub_end = ""
    contiguous_subs_list = list()
    srt_fix_list = list()

    # Faltten subtitles
    for i in range(len(srt_no_empties_list)-1):
        if srt.timedelta_to_srt_timestamp(srt_no_empties_list[i].end) == srt.timedelta_to_srt_timestamp(srt_no_empties_list[i+1].start):
            token_set_ratio = fuzz.token_set_ratio(srt_no_empties_list[i].content.lower().strip(), srt_no_empties_list[i+1].content.lower().strip())
            if token_set_ratio > 90:
                if sub_idx == "":
                    sub_idx = srt_no_empties_list[i].index
                if sub_start == "":
                    sub_start = srt_no_empties_list[i].start
                contiguous_subs_list.append(str(TextBlob(srt_no_empties_list[i].content).correct()).strip())
            else:
                srt_fix_list.append(srt_no_empties_list[i])
        else:
            if sub_idx == "":
                srt_fix_list.append(srt_no_empties_list[i])
            else:
                sub_end = srt_no_empties_list[i].end
                sub = srt.Subtitle(sub_idx, sub_start, sub_end, contiguous_subs_list[0])
                srt_fix_list.append(sub)
                sub_idx = ""
                sub_start = ""
                sub_end = ""
                contiguous_subs_list = list()

    srt_cps_list = list()
    for e in srt_fix_list:
        cps = characters_per_second(e.start, e.end, e.content)
        if cps < 50 and cps > 1:
            srt_cps_list.append(e)

    output = srt.compose(srt_cps_list)

    file_name = subs_path.split(".")[0] + "_fixed.srt"
    with open(file_name, "w", encoding="utf8") as f:
        f.write(output)


# subs_top_path = "video_top.srt"
# subs_bot_path = "video_bot.srt"
# subtitles_fixer(subs_top_path)
# subtitles_fixer(subs_bot_path)

# subs_top_path = "video_top_fixed.srt"
# subs_bot_path = "video_bot_fixed.srt"
# subtitles_joiner(subs_top_path, subs_bot_path)