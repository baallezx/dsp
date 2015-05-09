import scipy.io.wavfile as wav
from scipy.fftpack import fft , ifft
import time
import math
# -----  plot libraries  -----
from mpl_toolkits.mplot3d import Axes3D		# this is used for the polygon plots, seems like an older library
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np

#-- Possible error identification terms --
#
# @~error -- this means that there is a possible issue with n or (n-1) math

class Matrix:
	''' This contains both the signal, and frequency matrices for the wav file. '''
	def __init__(self):
		self.signal = []
		self.freq = []
		self.n = 0		# size of signal intervals to slice
		self.r = 0		# number of rows in the matrix
		self.padding = 0
		self.change_in_frequency = 0.05
		self.xplot = 0		# all of these plots are numpy array to be used for graphical representations
		self.yplot = 0
		self.zplot = 0
		# style tells which graph to plot
		self.style = 0
		self.strt_f = 0		# starting and ending coordinates for the frequency distribution
		self.stp_f = 0
		self.strt_t = 0
		self.stp_t = 0
		self.acc = 1

	def find_relevant_magnitudes(self,strt_r=0,stp_r=0,strt_c=0,stp_c=0):
		''' iterate over matrix with given coordinates replying with any relevant notes. '''
		if strt_r == 0 and stp_r == 0 and strt_c == 0 and stp_c == 0:
			#just iterate over the entire matrix
			highest_points = []
			row = 0
			for i in self.freq:
				p0 = 0
				count = 0
				pnt = 0
				for j in i:
					if j[2] > p0:
						p0 = j[2]
						pnt = count	# frequency ibtained at
						count += self.change_in_frequency
				highest_points.append([ row , count , p0 ])
				row += 1
			return highest_points
				
		else:
			pass
#	for i in matrix:
#		print i
#	for i in ui.freq:
#		print i
#
#	make_image(ui)

# edit this so that you can partition a matrix
	def get_nparray(self):
		''' this is used to return the appropriate numpy-arrays for use in plotting the data needed. '''
		x = []
		y = []
		z = []
		t0 = 0.0
		for i in self.freq:
			t = 0.0
			x0 = []
			y0 = []
			z0 = []
			for j in i:
				x0.append(t)
				y0.append(t0)
				z0.append(j[2])	#magnitude of the frequency at that location
				t += self.change_in_frequency
			t0 += self.change_in_frequency
			x0 = np.array(x0)
			y0 = np.array(y0)
			z0 = np.array(z0)
			x.append(x0)
			y.append(y0)
			z.append(z0)
		x = np.array(x)
		y = np.array(y)
		z = np.array(z)
		print 'x-type , x[0] = ',type(x),' , ',type(x[0]),'\ny-type , y[0] = ',type(y),' , ',type(y[0]),'\nz-type , z[0] = ',type(z),' , ',type(z[0])
		time.sleep(2)
		self.xplot = x
		self.yplot = y
		self.zplot = z

        def get_nparray_partition(self,strt_r,stp_r,strt_c,stp_c):
                ''' this is used to return the appropriate numpy-arrays for use in plotting the data needed.
	strt_r = row partition start point for matrix
	stp_r = row partition stop point for matrix 
	strt_c = column partition start point for matrix
	stp_c = column partition stop point for matrix'''
                x = []
                y = []
                z = []
                t0 = strt_r
                for i in xrange( strt_r , stp_r+1 ): #self.freq:
                        t = strt_r	#0.0
                        x0 = []
                        y0 = []
                        z0 = []
                        for j in xrange( strt_c , stp_c+1 ):
                                x0.append(t)
                                y0.append(t0)
                                z0.append(self.freq[i][j][2]) #magnitude of the frequency at that location
                                t += self.change_in_frequency
                        t0 += self.change_in_frequency
                        x0 = np.array(x0)
                        y0 = np.array(y0)
                        z0 = np.array(z0)
                        x.append(x0)
                        y.append(y0)
                        z.append(z0)
                x = np.array(x)
                y = np.array(y)
                z = np.array(z)
                print 'x-type , x[0] = ',type(x),' , ',type(x[0]),'\ny-type , y[0] = ',type(y),' , ',type(y[0]),'\nz-type , z[0] = ',type(z),' , ',type(z[0])
                time.sleep(2)
                self.xplot = x
                self.yplot = y
                self.zplot = z


	def show_wire_plot(self):
		if self.xplot is 0 or self.yplot is 0 or self.zplot is 0:
			print 'You first need to create the appropriate numpy arrays to graph your data!'
		else:
			# creates the window itself
			fig = plt.figure()
			if self.style == 0:	# wire-frame
				ax = fig.add_subplot(111, projection='3d')
#				X, Y, Z = axes3d.get_test_data(0.05)
				ax.plot_wireframe(self.xplot, self.yplot, self.zplot, rstride=self.acc, cstride=self.acc)
				ax.set_xlabel('Frequency')
				ax.set_ylabel('Time')
				ax.set_zlabel('Magnitude')
# test plotting multiple graphs
				if 0:
					px = fig.add_subplot(111,projection='3d')
					px.plot_wireframe(self.xplot, self.yplot, self.zplot, rstride=self.acc, cstride=self.acc)
					#px.set_zlim(0.0,20000000.0)
					#px.set_xlim(self.strt_t, self.stp_t)
					#px.set_ylim(self.strt_f, self.stp_f)
                                	px.set_xlabel('Time')
                        	        px.set_ylabel('Frequency')
                                	px.set_zlabel('Magnitude')
			elif self.style == 1:
				# add in the other graphs later
				pass
			print 'x = ',self.xplot,'\ny = ',self.yplot,'\nz = ',self.zplot
			time.sleep(2)
			plt.show()

	def get_verts(self, kind='signal'):
		''' This creates a list of appropriate points to feed into 3d polygon graph.  CURRENTLY NOT WORKING!!!!!!!!!!11 '''
		v = []
		t = 0
		if kind == 'signal':
			for i in self.signal:
				v2 = []
				for j in i:
# these may need to be tuples not lists
					v2.append((t,j))
					t += self.change_in_frequency
				v.append(v2)
				t = 0
			return v
		elif kind == 'freq':
# must do slightly different since this matrix is full of vectors [real , imag , magnitude ]. so you want the m[n][2] for each one, since we are graphing magnitude
			num_rows = 0.0
			for i in self.freq:
				v2 = []
				for j in i:
# these may need to be tuples not lists
					v2.append((t,j[2]))
					t += self.change_in_frequency
				v.append(v2)
				t = 0
				num_rows += 1.0
			return v
		else:
			print 'error in matrix construction!!!!!!!!'
			exit(1)
					

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
	for i in range(75):
		print '\n'#'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'

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
		if i.real < 0 and i.imag < 0:
			mag = math.sqrt((-1*i.real)**2+(-1*i.imag)**2)
		elif i.real < 0 and i.imag >= 0:
			mag = math.sqrt((-1*i.real)**2 + i.imag**2)
		elif i.real >= 0 and i.imag < 0:
			mag = math.sqrt(i.real**2+(-1*i.imag)**2)
		else:
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

def make_image(x):
	''' x wil be a matrix data structure. '''
	fig = plt.figure()
	ax = fig.gca(projection='3d')
#
	cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)
#
	xs = np.arange(0, 10, 0.4)
	verts = x.get_verts('freq')
	for i in xrange(17):
		print verts[0][i],'\n'
	for i in xrange(17):
		print verts[1][i],'\n'
	zs = [0.0, 1.0, 2.0, 3.0]
#	for z in zs:
#		ys = np.random.rand(len(xs))
#		ys[0], ys[-1] = 0, 0
#		verts.append(list(zip(xs, ys)))
#	poly = PolyCollection(verts, facecolors = [cc('r'), cc('g'), cc('b'),cc('y')])
	poly = PolyCollection(verts)
	poly.set_alpha(0.3)
#	ax.add_collection3d(poly, zs=zs, zdir='y')
	ax.add_collection3d(poly, zs=zs, zdir='y')
#
	ax.set_xlabel('X')
	ax.set_xlim3d(0, 123456)
	ax.set_ylabel('Y')
	ax.set_ylim3d(0, 65536)
	ax.set_zlabel('Z')
	ax.set_zlim3d(0, 1000)
#
	plt.show()


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
	ui.n = tc
	ui.signal = chop_sound(sound,tc)
	ui.r = len(ui.signal)
	ui.freq = get_fft(ui.signal)
	print 'len signl matrix = ',len(ui.signal),'\nlen frequency matrix = ',len(ui.freq)
# this part lets you visualize the matrix in either partitioned or non-partitioned
	userin = raw_input(str('Would you like to partition the M*N matrix where\n\nM = '+str(len(ui.signal[0]))+'\n\nand\n\nN = '+str(ui.r)+'\n\n--------------------\n\ny/n?   '))
	if userin == 'y':
		ui.acc = 1
		ui.strt_f = 0		# 0 
		ui.stp_f = 20		# 20
		ui.strt_t = 0		# 0
		ui.stp_t = 170		# 170
		# add interface for users to enter in partition amounts
		ui.get_nparray_partition(ui.strt_t,ui.stp_t,ui.strt_f,ui.stp_f)
	else:
		ui.get_nparray()
	ui.show_wire_plot()
	print ui.find_relevant_magnitudes()

if __name__ == '__main__':
	main()
