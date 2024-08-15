#VRMovieLab: A VIRTUAL REALITY PLATFORM FOR THE STUDY OF SCREEN MEDIA EFFECTS.
#Measuring physiological reponses to emotional films in a virtual laboratory.
#Author: Gary Bente, Michigan State University

#viz.link(viz.CenterCenter,circle)
import vizfx.postprocess
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
#possible key_vals for the physio target progranm
#Stop_value ="-" (ASCII 124)	#stop the recording
#silder_Values="0" bis"9"   (ASCII 48 bis 57) #slider value between 0(no value) 1 and 9 (scale -4 to +4)
#event_values ="A" bis "Z"  (ASCII 65 bis 90) #event makers to mark film segments
#either send by ord of str e.g. 1, 2 , 3 
#win32gui.PostMessage(hWnd,wmkeydown, ord(str(ev)),0)
#or by ascii keycode e.g.key_val=124
#win32gui.PostMessage(hWnd,wmkeydown, key_val,0)
wmkeydown=0x0100
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
	global plane
	#fl=open (fD+'_SCR.DAT','w')
	get_stimuli()
	#print(vlist)
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
		print(fiID)
		#print (t,vlist[t-1])		
		myVideo = viz.addVideo(fiID) 
		myBlank = viz.addVideo(blID) 
		#plane.texture(myVideo )
		
		w,h,d = myVideo.getSize()
		#print(w,h)
		aspectRatio = float(h) / float(w)
		quad.setPosition(-.25,0,4)
		sc=4.5 #3.5
		quad.setScale([sc,sc*aspectRatio, 1])		
		
		#quad.setScale([.8*w,.8*w*aspectRatio, 1])		
		quad.texture(myVideo)
		
		quad.visible(1)
		
		#ttV=myVideo.getDuration()
		#print((ttV))
		#ttV-=5.0

		stime=time.time()

		#time.sleep(3)
		zp=3.45
		yp=1.9
		if 'Relax' in fiID:
			canvas2.visible(0)#CRM
			instruct.message( emotext[0] )  #22 chars
			instruct.visible(1)
			xx=15  
			tt=66
		elif 'blank' in fiID:
			canvas2.visible(0)#CRM
			instruct.message( ' ')  #22 chars
			xx=10 #-.68+(.09*(22))
			instruct.visible(0)
			#instruct.setPosition(xx,yp,zp) #,
			tt=65
		else:
			z+=1
			if z==4:
				z=1
				zz+=1
			canvas2.visible(1)#CRM   #only if real stmulus not while relax
			instruct.message( emotext[zz])   #22 chars
			instruct.visible(1)
			
			xx=15  #-.68+(0.11*(27-len(emotext[zz])))
			#instruct.setPosition(xx,yp,zp) #,
			tt=t+66
		#instruct.setPosition(xx,190)
		#canvas3.visible(1)#Instruction
		quad.texture(myBlank)
		myBlank.play()
		sleep(2)
		quad.texture(myVideo)
		
		stime=time.time()
		event=65+(t)
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
		#quad.visible(0)
		ltime=time.time()
		fl_list=[]
		fl_list.append(vlist[t-1]) 		
		fl_list.append(str(stime)) 
		fl_list.append(str(etime)) 
		fl_list.append(str(ltime)) 
		data_list.append(fl_list)
		
		reset_CRM()
		

	s1=  124 #ord('-')
	win32gui.PostMessage(hWnd,wmkeydown, s1,0)

	df = pd.DataFrame (data_list, columns = ['Clip', 'start_movie','end_movie','end_rating','scary','funny','sad'])
	df.to_csv(fD+'_SCR.DAT', sep='\t', index=False)	
	#fl.close
	#time.sleep(1)
	viz.quit()
	

def reset_CRM():
	global slider_new
	global slider_old
	global t4a,t4b
	
	global instruct
	
	slider_old=5
	slider_new=5
	ofs=320 #65
	for i in range (1,10):
		circ[i] = vizshape.addCircle(parent=canvas2) #viz.addRadioButton(0,parent=canvas2)  
		circ[i].setScale([10.5,10,1])		
		circ[i].setPosition(ofs+50,i*25-74)
		circ[i].color(grey)
	circ[slider_old].color(yellow)

	t4a.setPosition(ofs+30,-74)
	t4b.setPosition(ofs+25,170)
	
	
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


def launch_Exp(ff):
	yield viztask.schedule( Video_2D(ff) )
	#Video_2D(ff)
	viz.window.setFullscreen(1)
	viz.mouse.setVisible(0)

def main():
	global alist
	global blist
	global mlist
	global vlist
	global button
	global canvas2
	global FilePHYS
	global FileSCR
	global FileID
	global N
	global r1
	global pair
	global tim
	global emotext
	global instruct
	global plane
	global quad
	global t4a,t4b

	#viz.MainView.setPosition([0,1,0])

	FileID = vizinput.input('Enter the Subject ID:')
	fID=respath+FileID

	gallery = vizfx.addChild('ground2.osgb')  #(

	quad = viz.addTexQuad(pos=[0,0,2])
	#quad.setScale([1.7,1.7*(9/16), 1])
	#quad.texture(video)
	quad.setReferenceFrame(viz.RF_VIEW)
	quad.disable(viz.DEPTH_TEST)
	quad.drawOrder(20)
	quad.visible(0)
	
	
	canvas1 = viz.addGUICanvas()
	canvas1.setRenderWorldOverlay([200,400],fov=40.0,distance=1)
	canvas1.setPosition([0,0,0])
	#canvas1.setMouseStyle(viz.CANVAS_MOUSE_VIRTUAL)
	canvas1.visible(1)
	canvas1.drawOrder(30)
	
	instruct= viz.addText(' ' ,parent=canvas1)   #22 chars
	instruct.font('ARIAL') 
	instruct.fontSize(16) 
	#instruct.setPosition(1,1,1)
	#instruct.setPosition(xx,yp,zp) #,
	instruct.setPosition(10,380)
			
	instruct.visible(0)
	emotext.append('Please try to relax!')
	emotext.append('How scary does it feel?')
	emotext.append('How funny does it feel?')
	emotext.append('How sad does it feel?')

		
	canvas2 = viz.addGUICanvas()
	canvas2.setRenderWorldOverlay([200,340],fov=40.0,distance=1)
	canvas2.setPosition([0.,.25,0])
	#canvas2.setMouseStyle(viz.CANVAS_MOUSE_VIRTUAL)
	canvas2.visible(0)
	
	t4a=viz.addText('not at all',parent=canvas2)
	t4b=viz.addText('very much',parent=canvas2)
	t4a.font('ARIAL') 
	t4a.fontSize(10) 
	t4b.font('ARIAL') 
	t4b.fontSize(10) 
	
	viz.setMultiSample(8) 
	viz.fov(40,16/9)
	viz.MainView.move([0,2.0,-2])
	

	reset_CRM()	
	
	
	
	#def UpdateTexture():
	#	monoQuad.texture(texture)

	#vizact.onupdate(0, UpdateTexture)


	def onButtonUp(e):
		#print(e.object, rControl)
		if e.object is rControl:
			#print('button',e.button,'up')
			if e.button==0:
				set_CRM('s')
			if e.button==1:
				set_CRM('a')

	def onMouseDown(button):
		if button == viz.MOUSEBUTTON_LEFT:
			set_CRM('s')

		elif button == viz.MOUSEBUTTON_RIGHT:
			set_CRM('a')

	viz.callback(viz.MOUSEDOWN_EVENT,onMouseDown)

	#viz.callback(viz.SENSOR_UP_EVENT,onButtonUp)
	#viz.callback(viz.KEYDOWN_EVENT,onKeyDown)
	
	headLight = viz.MainView.getHeadLight()
	headLight.enable()
	viz.go()

	viztask.schedule (launch_Exp(fID))




if __name__ == '__main__':
    main()

 

