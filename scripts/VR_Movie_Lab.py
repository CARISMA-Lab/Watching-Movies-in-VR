#VRMovieLab: A VIRTUAL REALITY PLATFORM FOR THE STUDY OF SCREEN MEDIA EFFECTS.
#Measuring physiological reponses to emotional films in a virtual laboratory.
#Author: Gary Bente, Michigan State University

import time
import pandas as pd
import viz
import vizfx
import vizact 
import vizinfo
import viztask
import vizconnect
import vizconfig
import vizshape
import vizmultiprocess
import time
from datetime import datetime
import vizinput
import logging
from random import *
import os
#import win32api
#import win32con
import sys
import os
import oculus
import random
import math
import win32gui
from time import sleep
global path
global respath
global mlist,vlist,alist,blist
global etime 
emotext=[]
#CONTROL OF THE PHYSIO PROGRAM
wmkeydown=0x0100
#possible key_vals for the physio target progranm
#Stop_value ="-" (ASCII 124)	#stop the recording
#silder_Values="0" bis"9"   (ASCII 48 bis 57) #slider value between 0(no value) 1 and 9 (scale -4 to +4)
#event_values ="A" bis "Z"  (ASCII 65 bis 90) #event makers to mark film segments
#either send by ord of str e.g. 1, 2 , 3 
#win32gui.PostMessage(hWnd,wmkeydown, ord(str(ev)),0)
#or by ascii keycode e.g.key_val=124
#win32gui.PostMessage(hWnd,wmkeydown, key_val,0)
hWnd = win32gui.FindWindow(None, 'PhysioTarget')

path=os.path.dirname(os.path.realpath(__file__))+'/'
path=path.replace('\\','/')
print(path)
respath=path+'results/'
stimpath=path+'stimuli/'

global slider_old
slider_old=5
global slider_new
slider_new=5
	
yellow=[1,1,0]
green=[0,1,0]
red =[1,0,0]
grey=[.1,.1,.1]

circ=[0,0,0,0,0,0,0,0,0,0,0]


		
def get_stimuli():
	
	global vlist
	vlist=[]
	fid1=stimpath+'Stimuli.txt'
	#fid1=stimpath+'test.txt'
	df = pd.read_csv(fid1,sep='\t',header=None)  #,usecols=lambda column : column in hvalid)
	
	#print(df)
	for i in range(len(df)):
		vlist.append (df.iloc[i,0])
	#print (df.iloc[i,0])
	#print(vlist)
	N=len(vlist)
	#random.shuffle(vlist)
	
def Video_2D(fD):
	global emotion
	global Angry	
	global Happy	
	global Sad
	global emotext
	global instruct
	global TV
	global pos1,pos2,pos3,pos4
	global vlist
	global path	

	#fl=open (fD+'_SCR.DAT','w')
	get_stimuli()
	
	yield viztask.waitKeyUp(' ')
	
	t=0
	N=len(vlist)
	#print(vlist)
	s1=  123 #ord('+')
	slider=48  #0
	event=65
	
	"""
	tx = viz.addText(' ',parent=canvas3)   #22 chars
	tx.font('ARIAL') 
	tx.fontSize(16) 
	tx.setPosition(515,700)	
	"""	
	win32gui.PostMessage(hWnd,wmkeydown, s1,0)
	win32gui.PostMessage(hWnd,wmkeydown, slider,0)
	win32gui.PostMessage(hWnd,wmkeydown, event,0)
	
	#myBlank= viz.addTexture(path+'Blank.jpg') 	
	data_list=[]
	z=0
	zz=1
	while t <  N:
		t=t+1
		fiID=stimpath+vlist[t-1]
		blID=stimpath+'blank.mp4'
		#print(fiID)
		#print (t,vlist[t-1])		
		myVideo = viz.addVideo(fiID) 
		myBlank = viz.addVideo(blID) 

		#ttV=myVideo.getDuration()
		#print((ttV))
		#ttV-=5.0

		stime=time.time()

		#time.sleep(3)

		if 'Relax' in fiID:
			canvas2.visible(0)#CRM
			instruct.message( emotext[0] )  #22 chars
			xx=-.68+(.09*(25-len(emotext[0])))
			instruct.setPosition(xx,2.16,3.45) #,
			tt=66
		elif 'blank' in fiID:
			canvas2.visible(0)#CRM
			instruct.message( ' ')  #22 chars
			xx=-.68+(.09*(22))
			instruct.setPosition(xx,2.16,3.45) #,
			tt=65
		else:
			z+=1
			if z==4:
				z=1
				zz+=1
			canvas2.visible(1)#CRM   #only if real stmulus not while relax
			#print(zz)
			#print (emotext[zz])
			
			instruct.message( emotext[zz])   #22 chars
			#g forinstruct.message( emotext[0])   #22 chars
			xx=-.68+(0.11*(27-len(emotext[zz])))
			instruct.setPosition(xx,2.16,3.45) #,
			tt=z+66

		#canvas3.visible(1)#Instruction
		
		TV.texture(myBlank)
		myBlank.play()

		sleep(2)
		myVideo = viz.addVideo(fiID) 
		TV.texture(myVideo )
		
		stime=time.time()
		event=tt
		win32gui.PostMessage(hWnd,wmkeydown, event,0)

		#myVideo.setTime(ttV)		
		myVideo.play()

		CUSTOM_EVENT = viz.getEventID("Ready")

		def onMedia(e):
			global etime,stime

			if  (e.object is myVideo and e.event == viz.MEDIA_END):
				etime=time.time()
				viz.sendEvent(CUSTOM_EVENT)
				event=65
				win32gui.PostMessage(hWnd,wmkeydown, event,0)
				

		viz.callback(viz.MEDIA_EVENT,onMedia)
				
		d=yield viztask.waitEvent(CUSTOM_EVENT)
		ltime=time.time()

		"""
		data = str(1-pos1)+'\t'+str(1-pos2)+'\t'+str(1-pos3)
		fl.write(fiID+'\t') 		
		fl.write(str(stime)+'\t') 
		fl.write(str(etime)+'\t') 
		fl.write(str(ltime)+'\t') 
		fl.write(data+'\n') 
		"""
		fl_list=[]
		fl_list.append(vlist[t-1]) 		
		fl_list.append(str(stime)) 
		fl_list.append(str(etime)) 
		fl_list.append(str(ltime)) 
		fl_list.append(str(round(1-pos1,2)))
		fl_list.append(str(round(1-pos2,2)))
		fl_list.append(str(round(1-pos3,2)))
		
		data_list.append(fl_list)
		
		reset_CRM()
		

	s1=  124 #ord('-')
	win32gui.PostMessage(hWnd,wmkeydown, s1,0)

	df = pd.DataFrame (data_list, columns = ['Clip', 'start_movie','end_movie','end_rating','scary','funny','sad'])
	df.to_csv(fD+'_SCR.DAT', sep='\t', index=False)	
	#fl.close
	#time.sleep(1)
	viz.quit()
	
	
def reset_radio_buttons():
	#Happy.set(viz.OFF)
	#Angry.set(viz.OFF)
	#Sad.set(viz.OFF)

	cx=0

def sliderChanged1(pos):
#    print 'certainty is at' + str(pos)
	global pos1
	pos1=pos

def sliderChanged2(pos):
#    print 'arousal is at' + str(pos)
	global pos2
	pos2=pos

def sliderChanged3(pos):
#    print 'valence is at' + str(pos)
	global pos3	
	pos3=pos
	
def sliderChanged4(pos):
#    print 'valence is at' + str(pos)
	global pos4	
	pos4=pos	

def onKeyDown(key):
	global slider_new
	global slider_old

	if key=='a':
		if slider_new==9:
			return
		else:
			slider_new+=1
	elif key=='s':	
		if slider_new==1:
			return
		else:	
			slider_new-=1 
			
	for i in range (1,10):
		circ[i].color(grey)
	circ[5].color(yellow)		
	
	
#	print(slider_new)
	if slider_new ==5:
		v=0
	if slider_new >5:
		for i in range (6,slider_new+1):
			circ[i].color(green)					
	if slider_new <5:
		for i in range (4,slider_new-1,-1):
			circ[i].color(red)					

	win32gui.PostMessage(hWnd,wmkeydown, slider_new+48,0)

def set_CRM(k):
	global slider_new
	global slider_old

	if k=='a':
		if slider_new==9:
			return
		else:
			slider_new+=1
	elif k=='s':	
		if slider_new==1:
			return
		else:	
			slider_new-=1 
			
	for i in range (1,10):
		circ[i].color(grey)
	circ[5].color(yellow)		
	
	
#	print(slider_new)
	if slider_new ==5:
		v=0
	if slider_new >5:
		for i in range (6,slider_new+1):
			circ[i].color(green)					
	if slider_new <5:
		for i in range (4,slider_new-1,-1):
			circ[i].color(red)					

	win32gui.PostMessage(hWnd,wmkeydown, slider_new+48,0)


def onButton(obj,state):  
	return
	global emotion
	global Angry	
	global Happy	
	global Sad
	if obj == r1:
		emotion=0
		
	if obj == Angry:
		emotion=1
	if obj == Happy:
		emotion=2
	if obj == Sad: 
		emotion=3

		
		
def onMouseMove(e): 
	#e.x, 'is absolute x.' 
	#e.y, 'is absolute y.' 
	#e.dx, 'is the relative change in x.' 
	#e.dy, 'is the relative change y.' 
	
	x=e.x
	y=e.y
	canvas.setCursorPosition ([x,y],mode = viz.CANVAS_CURSOR_NORMALIZED)
	fc=canvas.getCursorPosition (mode = viz.CANVAS_CURSOR_NORMALIZED)


def launch_Exp(ff):
	yield viztask.schedule( Video_2D(ff) )
	#Video_2D(ff)
	
def remap(x,inMin,inMax,outMin,outMax):
		return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin

def reset_CRM():
	global slider_new
	global slider_old
	slider_old=5
	slider_new=5
	for i in range (1,10):
		circ[i] = vizshape.addCircle(parent=canvas2) #viz.addRadioButton(0,parent=canvas2)  
		circ[i].setScale([25,17,1])		
		circ[i].setPosition(1070,145 + (i-1)*40)
		circ[i].color(grey)
	circ[slider_old].color(yellow)

def main():

	global alist
	global blist
	global mlist
	global vlist
	global An
	global Angry
	global Ha
	global Happy
	global Sa
	global Sad
	global ava
	global button
	global canvas
	global canvas2
	global canvas3
	global datHMD
	global FileHMD
	global FilePHYS
	global FileSCR
	global TV
	global emotion
	global mySlider1,mySlider2,mySlider3,mySlider4
	global pos1,pos2,pos3,pos4
	global t0
	global t1,t1a,t1b
	global t2,t2a,t2b
	global t3,t3a,t3b
	global t4,t4a,t4b
	global N
	global r1
	global pair
	global tim
	global emotext
	global instruct
	datHMD=[]
	
	pos1=.5
	pos2=.5
	pos3=.5
	pos4=.5
	emotion=0
	
	viz.setMultiSample(8) 
	viz.fov(40,16/9)
	viz.MainView.move([0,-5.0,0])
	#viz.MainView.setPosition([0,1,0])
	
	FileID = vizinput.input('Enter the Subject ID:')
	fID=respath+FileID
	
	
	#from here Oculus

	# Setup Oculus Rift HMD

	hmd = oculus.Rift()
	if not hmd.getSensor():
		sys.exit('Oculus Rift not detected')

	supportPositionTracking = hmd.getSensor().getSrcMask() & viz.LINK_POS

	# Setup heading reset key

	# Check if HMD supports position tracking

	#if supportPositionTracking:

		# Add camera bounds model
	#	camera_bounds = hmd.addCameraBounds()
		#camera_bounds.visible(OFF)

		# Change color of bounds to reflect whether position was tracked
	#	def CheckPositionTracked():
	#		if hmd.getSensor().getStatus() & oculus.STATUS_POSITION_TRACKED:
	#			camera_bounds.color(viz.GREEN)

	#vizact.onupdate(0, CheckPositionTracked)
	
	# Setup navigation node and link to main view


	#has to go in again in for haedset

	navigationNode = viz.addGroup()
	viewLink = viz.link(navigationNode, viz.MainView)
	viewLink.preMultLinkable(hmd.getSensor())


	profile = hmd.getProfile()
	#print(profile)
	# Apply user profile eye height to view

	"""
	if profile:
		navigationNode.setPosition([0,profile.eyeHeight,0])
	else:
		x=x
	"""
	
	navigationNode.setPosition([0,1.20,0])

	# Setup arrow key navigation
	MOVE_SPEED = 2.0
	
	
	def UpdateView():
		yaw,pitch,roll = viewLink.getEuler()
		#x,y,z = viewLink.getPosition()
		m = viz.Matrix.euler(yaw,0,0)
		dm = viz.getFrameElapsed() * MOVE_SPEED
		navigationNode.setPosition(m.getPosition(), viz.REL_PARENT)
		#print yaw,pitch,roll,x,y,z

	#up to here Oculus
	gallery = vizfx.addChild('LivingRoom_new.osgb') #('ground.osgb')
	TV=gallery.getChild('screen2', flags = viz.CHILD_DEFAULT)
	speaker=gallery.getChild('speaker', flags = viz.CHILD_DEFAULT)
	#print(TV)
	#DEll Laptop screen dimensions
	x=1920 #3840 
	y=1080 #2160
	
	#Create CANVAS a static 2D overlay to 3D scene
	canvas = viz.addGUICanvas()
	canvas.setRenderWorld([1400,1120],[7,3.5],scene=viz.MainScene)
	canvas.setPosition(-2.9,-.02,3.4)
	canvas.setMouseStyle(viz.CANVAS_MOUSE_VIRTUAL)
	canvas.visible(0)
	
	pos=[.13,.86,3.4]

	canvas2 = viz.addGUICanvas()
	canvas2.setRenderWorld([3000,1100],[7,3.5],scene=viz.MainScene)
	canvas2.setPosition([-1.15,.48,3.5])
	canvas2.setMouseStyle(viz.CANVAS_MOUSE_VIRTUAL)
	canvas2.visible(0)


	#CRM=gallery.getChild('CRM', flags = viz.CHILD_DEFAULT)
	reset_CRM()	
	#Placing all GUI Elements on CANVAS
	
	
	#fontsizes
	f1=18
	f2=16
	#Create an invisible Radio button to be the default selected one
	#so that one of the visible ones must be chosen by the user
	
	emotext.append('Please try to relax!')
	emotext.append('How scary does it feel?')
	emotext.append('How funny does it feel?')
	emotext.append('How sad does it feel?')
	
	instruct= viz.addText3D(' ',pos=(0,0,0)) #,parent=speaker)   #22 chars
	#instruct.alignment(viz.ALIGN_CENTER_BOTTOM)
	#instruct.disable(viz.LIGHTING)
	instruct.font('ARIAL') 
	instruct.fontSize(.08) 

	t0 = viz.addText('What was the emotional tone of the movie?',parent=canvas)   #22 chars
	t0.font('ARIAL') 
	t0.fontSize(16) 
	t0.setPosition(450,700)	
	
	
	
	t1 = viz.addText('scary',parent=canvas)
	t1.font('ARIAL') 
	t1.fontSize(f1) 
	t1a = viz.addText('no',parent=canvas)
	t1a.font('ARIAL') 
	t1a.fontSize(f2) 
	t1b= viz.addText('yes',parent=canvas)
	t1b.font('ARIAL') 
	t1b.fontSize(f2) 

	t2 = viz.addText('funny',parent=canvas)
	t2.font('ARIAL') 
	t2.fontSize(f1) 
	t2a = viz.addText('no',parent=canvas)
	t2a.font('ARIAL') 
	t2a.fontSize(f2) 
	t2b= viz.addText('yes',parent=canvas)
	t2b.font('ARIAL') 
	t2b.fontSize(f2) 

	t3 = viz.addText('sad',parent=canvas)
	t3.font('ARIAL') 
	t3.fontSize(f1) 
	t3a = viz.addText('no',parent=canvas)
	t3a.font('ARIAL') 
	t3a.fontSize(f2) 
	t3b = viz.addText('yes',parent=canvas)
	t3b.font('ARIAL') 
	t3b.fontSize(f2) 



	t4a=viz.addText('not at all',parent=canvas2)
	t4b=viz.addText('very much',parent=canvas2)
	t4a.font('ARIAL') 
	t4a.fontSize(18) 
	t4b.font('ARIAL') 
	t4b.fontSize(18) 
	
	t4a.setPosition(1040,140-40)
	t4b.setPosition(1030,140+360)
	
	
	alist=[]
	blist=[]
	vlist=[]
	atlist=[]
	btlist=[]
	vtlist=[]
	mlist=[]
	tlist=[]
	pair=[]
	N=0
#####Mouse stuff for VR
	rControl = hmd.getRightTouchController()
	
	def onButtonUp(e):
		#print(e.object, rControl)
		if e.object is rControl:
			#print('button',e.button,'up')
			if e.button==0:
				set_CRM('s')
			if e.button==1:
				set_CRM('a')

	
	viz.callback(viz.SENSOR_UP_EVENT,onButtonUp)
	
	viz.callback(viz.BUTTON_EVENT,onButton) 
	viz.callback(viz.MOUSE_MOVE_EVENT,onMouseMove)
	viz.callback(viz.KEYDOWN_EVENT,onKeyDown)
	
	viz.go()

	
	### add HMD
	FileHMD =respath + FileID+'_HMD.DAT'
	
	def get_HMD_data():
		roll,pitch,yaw=viewLink.getEuler()
		x,y,z=viewLink.getPosition()
		lt=time.time()  #datetime.now().time()
		dat= str(lt)+'\t'+str(round(x,4))+'\t'+str(round(y,4))+'\t'+str(round(z,4))+'\t'+str(round(roll))+'\t'+str(round(pitch,4))+'\t'+str(round(yaw,4))
		File = open(FileHMD, 'a')
		File.write("%s\n" % dat)
		File.close
		
		#datHMD.append (dat)
		#print dat   #x,y,z,roll,pitch,yaw
	
	vizact.ontimer(.02,get_HMD_data)
		
	
	viztask.schedule (launch_Exp(fID))

	

if __name__ == '__main__':
    # This is the main entry point of the program.
    # This will not be executed by child processeslaunched by vizmultiprocess.
    main()