#source: https://github.com/federico-terzi/bipcut/blob/master/bipcut.py  author Federico Terzi
#modified by Sean Benson

# MIT License
# 
# Copyright (c) 2018 Federico Terzi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import division, print_function
from scipy.io import wavfile
import numpy as np
import more_itertools as mit
import subprocess, tempfile, sys, os, os.path
import logging
from unyt import s

#######################################################################################
# CONFIGURATION PARAMETERS

# Configure the path to FFMPEG
# NOTE: if ffmpeg is available in the PATH environment varable, leave this None
FFMPEG_PATH = None

# Configure which frequencies BIPCUT should recognize
# NOTE: if you add another frequency, you should add it to TARGET_FREQS as well
START_CLIP_FREQ = 2000    # Frequency to Start a new clip ( and confirm the previous ) ( Hertz )
ERROR_CLIP_FREQ = 2600    # Frequency to discard the last clip ( Hertz )

# These are the frequencies that will be checked
TARGET_FREQS = (START_CLIP_FREQ, ERROR_CLIP_FREQ)

# The duration of the BEEP in seconds
BEEP_DURATION = 0.4
#######################################################################################

class Bipcut:
	#Author Federico Terzi
	# Analyze the audio to get the time markers
	def analyze_audio(self, audio_filename, target_freq=TARGET_FREQS, win_size=5000, step=200, min_delay=BEEP_DURATION, sensitivity=250, verbose=True):
		"""
		Analyze the given audio file to find the tone markers, with the respective frequency and time position.
		:param str audio_filename: The Audio filename to analyze to find the markers.
		:param tuple target_freq: A tuple containing the int frequencies ( in Hertz ) that the function should recognize.
		:param int win_size: The size of the moving window for the analysys.
							Increasing the window increases the accuracy but takes longer.
		:param int step: the increment between each window.
		:param float min_delay: Minimum duration, in seconds, of the beep to be recognized.
		:param int sensitivity: Minimum value of relative amplitude of the beep to be recognized.
		:param bool verbose: If true, print some info on the screen.
		:return: a list of dict containing the markers positions and frequencies.
		"""
		logging.debug("Making timemarkers by analyzing the audio...")
		print("Analyzing the Audio...")

		# Open the wav audio track
		# Get the sample rate (fs) and the sample data (data)
		fs, data = wavfile.read(audio_filename)

		# Calculate the duration, in seconds, of a sample
		sample_duration = 1.0 / fs

		# Get the total number of samples
		total_samples = data.shape[0]

		# Calculate the frequencies that the fourier transform can analyze
		frequencies = np.fft.fftfreq(win_size)
		# Convert them to Hertz
		hz_frequencies = frequencies * fs

		# Calculate the indexes of the frequencies that are compatible with the target_freq
		freq_indexes = []
		for freq in target_freq:
			# Find the index of the nearest element
			index = (np.abs(hz_frequencies - freq)).argmin()
			freq_indexes.append(index)

		# This will hold the duration of each frequency pulse
		duration_count = {}
		# Initialize the dictionary
		for freq in target_freq:
			duration_count[freq] = 0

		# Initialize the counter
		count = 0

		# This list will hold the analysis result
		results = []

		# Analyze the audio dividing the samples into windows, and analyzing each
		# one separately
		for window in mit.windowed(data, n=win_size, step=step, fillvalue=0):
			# Calculate the FFT of the current window
			fft_data = np.fft.fft(window)

			# Calculate the amplitude of the transform
			fft_abs = np.absolute(window)

			# Calculate the mean of the amplitude
			fft_mean = np.mean(fft_abs)

			# Calculate the current time of the window
			ctime = count * sample_duration

			# Check, for each target frequency, if present
			for i, freq in enumerate(target_freq):
				# Get the relative amplitude of the current frequency
				freq_amplitude = abs(fft_data[freq_indexes[i]]) / fft_mean

				# If the amplitude is greater than the sensitivity,
				# Increase the duration counter for the current frequency
				if freq_amplitude > sensitivity:
					duration_count[freq] += step * sample_duration
				else:
					# If the duration is greater than the minimum delay, add the result
					if duration_count[freq] > min_delay:
						results.append({'time': ctime, 'freq': freq})

						# Print the result if verbose
						if verbose:
							print("--> found freq:", freq, "time:", ctime)
					duration_count[freq] = 0
			count += step

			# Print the progress every 100000 samples
			if verbose and count % 100000 == 0:
				percent = round((count/total_samples) * 100)
				print("\rAnalyzing {}% ".format(percent), end="")

		print()  # Reset the new line
		return results

	#Author Federico Terzi
	# Get the path for the FFMPEG executable
	def get_ffmpeg_path(self):
		"""
		Get the FFMPEG path. Check if FFMPEG is available in the PATH environment variable,
		if it isn't, it tries in the FFMPEG_PATH variable.
		:return: the FFMPEG path as string 
		"""
		try:
			# Check if FFMPEG is available in the PATH environment variable
			p = subprocess.Popen(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			out, err = p.communicate()
			return "ffmpeg"
		except Exception as e:
			pass

		# Try with the FFMPEG_PATH variable
		if FFMPEG_PATH is not None:
			try:
				p = subprocess.Popen([FFMPEG_PATH, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				out, err = p.communicate()
				return FFMPEG_PATH
			except Exception as e:
				pass

		print("FFMPEG could not be found, please add it to the PATH variable or change the FFMPEG_PATH variable in the BIPCUT script")
		sys.exit(1)

	#Author Federico Terzi
	# Extract the audio track
	def ffmpeg_extract_audio(self, ffmpeg_path, original_file):
		"""
		Extract the Audio track from the original file into a temporary file using FFMPEG
		:return: the path to the audio track temporary file
		"""
		logging.debug("analyizing audio")
		# Create a temporary output file
		output_file = tempfile.gettempdir() + os.sep + "bipcut_tempfile.wav"

		print("Extracting the audio track...")

		# Execute ffmpeg to extract the audio track
		p = subprocess.Popen([ffmpeg_path, "-y", "-i", original_file, "-ac", "1", output_file])
		return_code = p.wait()

		# Make sure the ffmpeg executed correctly by checking the exit code
		if return_code != 0:
			print("Can't extract the audio track!")
			sys.exit(2)

		return output_file

	#Author Federico Terzi
	def ffmpeg_extract_clip(self, ffmpeg_path, original_file, output_file, start, stop):
		"""
		Extract the clip between start and stop, from the given file to the output_file, using FFMPEG.
		:return: True if succeeded, false otherwise.
		"""
		#ffmpeg -ss 4.63482993197 -to 14.8733106576 -i "D:\\Dropbox\\Caricamenti da fotocamera\\rec.wav" output.mp3

		# Execute ffmpeg to extract the clip
		p = subprocess.Popen([ffmpeg_path, "-y", "-i", original_file, "-c", "copy", "-ss", str(start), "-to", str(stop), output_file])
		return_code = p.wait()

		# Make sure the ffmpeg executed correctly by checking the exit code
		if return_code != 0:
			return False

		return True

	#Author Sean Benson
	#Checks the inputed string which is the file name to see if it exist.
	def check_file_existance(self, input_file):
		if not os.path.isfile(input_file):
			logging.debug("Input file is not valid!")
			sys.exit(4)
		else:
			logging.debug("Input file: " + input_file)

	#Author Sean Benson
	#Checks the inputed string which is the file directory to see if it exist.
	def check_directory_existance(self, output_directory):
		if not os.path.isdir(output_directory):
			logging.debug("Output directory is not valid!")
			sys.exit(5)
		else:
			logging.debug("Output Directory: " + output_directory)

	#Author Sean Benson
	def extract_beep_clips(self, input_file, output_directory, output_format, ffmpeg_path, time_markers):
		for time_marker in time_markers:
			self.extract_beep_clip(time_marker, input_file, output_directory, output_format, ffmpeg_path)
		
	#author Sean Benson
	def extract_beep_clip(self, time_marker, input_file, output_directory, output_format, ffmpeg_path):
		# Initialize the ranges
		start_time = 0
		end_time = 0
		# Check the meaning of the marker, and execute the corresponding action
		if time_marker['freq'] == START_CLIP_FREQ:
			print("START AT:", time_marker['time'])
			# Confirming the last segment
			# Update the end time and remove the beep sound duration
			end_time = time_marker['time'] - BEEP_DURATION - 0.2

			# Create the clip filename
			clip_filename = os.path.basename(input_file) + "." + str(round(time_marker['time'])) + "." + output_format
			clip_path = os.path.join(output_directory, clip_filename)

			# Extract the clip
			self.ffmpeg_extract_clip(ffmpeg_path, input_file, clip_path, start_time, end_time)
			print("Extracted clip between START:", start_time, "END:", end_time)
				
			# Reset the start time
			start_time = time_marker['time']
		elif time_marker['freq'] == ERROR_CLIP_FREQ:
			print("ERROR AT: ", time_marker['time'])
			start_time = time_marker['time']
		else:
			print("UNKNOWN FREQUENCY: ", time_marker['freq'])

	#author Sean Benson
	def extract_clip(self, beginning_time, ending_time, input_file, output_directory, output_format, ffmpeg_path):
		# Create the clip filename
		clip_filename = os.path.basename(input_file) + "." + str(beginning_time) + "." + output_format
		clip_path = os.path.join(output_directory, clip_filename)
		self.ffmpeg_extract_clip(ffmpeg_path, input_file, clip_path, beginning_time.value, ending_time.value)
		print("Extracted clip between START:", beginning_time, "END:", ending_time)

	#Authors Fedrico Terzi modified by Sean Benson
	#Extract all the given clips from the markers
	def start_extracting_beep_clips(self, input_file, output_directory, output_format):
		self.check_file_existance(input_file)
		self.check_directory_existance(output_directory)
		ffmpeg_path = self.get_ffmpeg_path()   
		audio_track_file = self.ffmpeg_extract_audio(ffmpeg_path, input_file)  
		time_markers = self.analyze_audio(audio_track_file)
		self.extract_beep_clips(input_file, output_directory, output_format, ffmpeg_path, time_markers)
		os.remove(audio_track_file)  #Delete the temporary file
		sys.exit(0)  #Quit python

	#Authors Fedrico Terzi modified by Sean Benson
	#Extract all the given clips from the markers
	def start_extracting_random_clip(self, input_file, output_directory, output_format):
		self.check_file_existance(input_file)
		self.check_directory_existance(output_directory)
		ffmpeg_path = self.get_ffmpeg_path()   
		audio_track_file = self.ffmpeg_extract_audio(ffmpeg_path, input_file)  
		beginning_time = 3*s
		ending_time = 10*s
		self.extract_clip(beginning_time, ending_time, input_file, output_directory, output_format, ffmpeg_path)
		os.remove(audio_track_file)  #Delete the temporary file
		sys.exit(0)  #Quit python