import scipy.io.wavfile as wav
from scipy.fftpack import fft,ifft
import time
import math

#-- Possible error identification terms --
#
# @~error -- this means that there is a possible issue with n or (n-1) math

class Matrix:
	''' This contains both the signal, and frequency matrices for the wav file. '''
	def __init__(self):
		self.signal = []
		self.freq = []

#	def load_signal_matrix(self):
#		self.signal = 
		

def split_stereo(x,side):
	''' x needs to be the array only part of the wave file \nside = returned array of (left or right) '''
	l = []
	if side == 'left':
		for i in xrange(len(x)):
			l.append(x[i][0])
	elif side == 'right':
		for i in xrange(len(x)):
			l.append(x[i][1])
	return l

def read_file(filename):
	# a = sample rate
	# b = data array
	# d = data type 'ex. 16 bit integer'
	(a,b) = wav.read(filename)
	d = str(b.dtype)
	return a,b,d

def clr_scr():
	for i in range(255):
		print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'

def find_padding(n):
	''' iterate through powers of 2 until you find the closest power of 2 that is greater than n, and then subtract the numbers for the padding amount.  '''
	found_power = False
	i = 0
	while not found_power:
		po2 = 2**i
		i+=1
		if po2 > n:
			found_power = True
	print '\n\npadding amount = ',(po2-n)
	# return the padding amount
	return po2 - n

#??????????????????????????????
def chop_sound(s,n):
	''' s = singal , n = interval to chop at. returns matrix which has all of the chooped signals in their coresponding rows to be run through the FFT. '''
	matrix = []
	# check to see if n is a power of 2 if not give info for necessary padding
	power = power_of_2(n)
	padding = 0
	if not power:
		padding = find_padding(n)
	len_s = len(s)
	# python rounds down, so in case of a decimel you add 1
	if len_s % n == 0:
		chop_point = len_s/n
	else:
		chop_point = (len_s/n)+1
	curr_signal_pos = 0
	curr_chop_point = 0
	pts_used = (chop_point-1)*n
	pts_r = len_s - pts_used
#------------------------------------------------------
	for i in s:
		curr_signal_pos+=1
	print '\n\n\n\ncurrent signal position = ',curr_signal_pos
	curr_signal_pos = 0
	time.sleep(5)
#------------------------------------------------------
	for i in xrange(chop_point):
		row = []
# need to check to see if this is the final part to chop
# this needs to be changed from 2 for-loops inside an if block to 1 for-loop and an if block inside.
		if curr_chop_point == chop_point - 1:
			# take care of final array
			for j in xrange((n + padding)):
				if j > (pts_r - 1):
					row.append(0)
				else:
					row.append(s[curr_signal_pos])
					curr_signal_pos += 1
#					print '\nadded to final row'
			matrix.append(row)
			curr_chop_point += 1
		else:
			for j in xrange((n + padding)):
# @~error
				if j > (n-1):
					row.append(0)
				else:
					row.append(s[curr_signal_pos])
		# only increment curr_signal_pos in this part. if you increment anywhere else you will be picking out of order 
					curr_signal_pos += 1
					#print curr_signal_pos
			matrix.append(row)
			curr_chop_point += 1
	return matrix

def power_of_2(num):
	''' check to see if number is power of 2. If NOT then return padding necessary for each array. '''
        while (num % 2) == 0 and num > 1:
                num /= 2
        return num == 1

def convert_complex(row):
	''' new rows consist of [ [ real_0 , imag_0 , magnitude_0 , phase_0 ], ... , [ real_n , imag_n , magnitude_n , phase_n ] '''
	rr = []
	# need to come up with a method to give the phase its appropriate quadrant.
	for i in row:
		mag = math.sqrt(i.real**2 + i.imag**2)
		if mag < 0:
			mag = -1*mag
		rr.append([i.real , i.imag , mag]) #, math.atan(i.imag/i.real)])
	return rr

def get_fft(matrix):
	''' return a matrix form frequency grid from an input of a signal matrix. '''
	fm = []
	for i in matrix:
		fm_row = fft(i)
		rr1 = convert_complex(fm_row)
		fm.append(rr1)
	return fm

def main():
	clr_scr()
	#restart = False
	userin = raw_input('Please enter in the name of the file that you want to use:  ')
	try:
		a,b,d = read_file(userin)
	except:
		clr_scr()
		print '\n\n\nYou have entered in an incorrect filename. Please check your spelling and try again...'
		time.sleep(2)
		exit(1)
	clr_scr()
	userin = raw_input('Would you like to do a fourier transform on the left, right, or convert you track into a mono file?\n---------------------------\n1 - left\n2 - right\n3 - mono\n4 - exit\n----------------------------\n')
	sound = []
	ext = True
	if userin == '1':
		sound = split_stereo(b,'left')
	elif userin == '2':
		sound = split_stereo(b,'right')
	elif userin == '3':
		# at this time I am only going to return the left side of the track. 
		# This needs to be changed since panning and other effects use the 
		# movement between speakers heavily.
		sound = split_stereo(b,'left')
	else:
		clr_scr()
		ext = False
		print 'Goodbye!!!'
	if ext:
		print 'you have a sound file and its length is ', len(sound),'.'
	userin = raw_input('At what interval do you want to cut the track: ')
	tc = ''
	while type(tc) is not int:
		try:
			tc = int(userin)
		except:
			print 'Please enter in a number between 4 and ',len(sound),'!'
			userin = raw_input('enter in cut interval:  ')
	if tc < 4:
		print 'Sorry you need to pick a number between 4 and ',len(sound),'!'
		exit(1)
	ui = Matrix()
	ui.signal = chop_sound(sound,tc)
	ui.freq = get_fft(ui.signal)
	print 'len signl matrix = ',len(ui.signal),'\nlen frequency matrix = ',len(ui.freq)
#	for i in matrix:
#		print i
	for i in ui.freq:
		print i

if __name__ == '__main__':
	main()
