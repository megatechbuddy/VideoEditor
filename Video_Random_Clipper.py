from Video_Modifier_Core import Video_Modifier_Core
import os, sys

#######################################################################################
# CONFIGURATION PARAMETERS

input_file = 'D:/Videos/editing/file2.mp4'   #sys.argv[1]
output_directory = 'D:/Videos/editing'   #sys.argv[2]
output_format = 'mp4'   #sys.argv[3]

#######################################################################################
# Methods and start

video_Modifier_Core = Video_Modifier_Core.Bipcut()
video_Modifier_Core.start_extracting_random_clip(input_file, output_directory, output_format)

