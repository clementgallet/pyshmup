import random
import math
from constants import *
import OpenGL.GL as gl


class Shot:

	def __init__(self, context):

		self.x = 0
		self.y = 0
		self.lines = []

		context.shot_list.append(self)
		context.update_list.append(self)
		
		self.to_remove = False
		self.aimed_foe = None
		
		self._context = context
		
	def update(self):

		if not self.aimed_foe in self._context.foe_list and not self.to_remove:
			self.vanish()
			return self

		dist = (self.x - self.aimed_foe.x)**2 + (self.y - self.aimed_foe.y)**2

		xpos = ((self.aimed_foe.x > self.x) + (self.aimed_foe.x >= self.x) - 1)*(((self.aimed_foe.x - self.x)**2)/dist) + 1
		xneg = 2 - xpos
		ypos = ((self.aimed_foe.y > self.y) + (self.aimed_foe.y >= self.y) - 1)*(((self.aimed_foe.y - self.y)**2)/dist) + 1
		yneg = 2 - ypos

		xpos *= xpos
		xneg *= xneg
		ypos *= ypos
		yneg *= yneg
		
		choix = random.random()*(xpos + xneg + ypos + yneg)

		shot_dist = math.sqrt(dist)/2
		
		if choix < xpos:
			self.x += shot_dist
		elif xpos <= choix < xneg + xpos:
			self.x -= shot_dist
		elif xneg + xpos <= choix < xneg + xpos + ypos:
			self.y += shot_dist
		else:
			self.y -= shot_dist

		self.lines.append((self.x,self.y))

		return self
	
	def draw(self):

		if len(self.lines) > NB_LINES:
			self.lines.pop(0)
			
		taille = len(self.lines) - 1
		c = 0

		if taille >= 0:
			gl.glDisable(gl.GL_TEXTURE_2D)
			for i in self.lines:
				(x,y) = i
				try:
					pen_x = pen_x
					SHOT_COLOR[3] = 1 - (float(taille - c) / NB_LINES)
					gl.glBegin(gl.GL_LINES)
					gl.glColor4f(*SHOT_COLOR)
					gl.glVertex2f(x, y)
					gl.glVertex2f(pen_x, pen_y)
					gl.glEnd()
				except:
					pass
				
				pen_x = x
				pen_y = y
				
				c += 1
			gl.glColor4f(1.0, 1.0, 1.0, 1.0)


	def vanish(self):
		self._context.shot_list.remove(self)
		self.to_remove = True
