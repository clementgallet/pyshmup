from constants import *

import OpenGL.GL as gl
import math
import pygame
import coll
import random
import sprite

from shot import Shot

class Player:
	def __init__(self, context):
		self._context = context

		self.x = 0.0
		self.y = -UNIT_HEIGHT * .5
		self.frame = 0
		self.id = len(context.player_list) # Won't work when a death occurs
		context.update_list.append(self)
		context.player_list.append(self)
		self.to_remove = False
		self.speed = PLAYER_SPEED

		self.sprite = sprite.get_sprite(SHIP_BITMAP)

		i = 1
		while gl.glIsList(i):
			i += 1
		gl.glNewList(i, gl.GL_COMPILE)
		
		gl.glColor4f(1.0, 0.0, 1.0, 0.5)
		gl.glDisable(gl.GL_TEXTURE_2D)
		gl.glBegin(gl.GL_TRIANGLE_FAN)
		NB_STRIPS = 32
		gl.glVertex2f(0., 0.)
		for k in xrange(NB_STRIPS+1):
			gl.glVertex2f(RADIUS * math.cos(2 * math.pi * k / NB_STRIPS),
			           RADIUS * math.sin(2 * math.pi * (-k) / NB_STRIPS))
		gl.glEnd()
		gl.glColor4f(1., 1., 1., 1.)

		gl.glEndList()
		self.circle_list = i

		self.t=0

	def update(self):
		dx = 0
		dy = 0
		keys = self._context._system_state.inputs[self.id]
		if keys[INDEX_RIGHT]:
			dx += 1
		if keys[INDEX_LEFT]:
			dx -= 1
		if keys[INDEX_UP]:
			dy += 1
		if keys[INDEX_DOWN]:
			dy -= 1
		self.x += dx*PLAYER_SPEED
		self.y += dy*PLAYER_SPEED
		self.frame += 1
		
		for i in range(self._context.array_fill - 1, -1, -1):
			if (self._context.bullet_array[i,ARRAY_COLLIDE_MASK] & 1 << self.index):
				self._context.delete_bullet(i)
				self.to_remove = True
		
		for i in range(self._context.array_ml_fill - 1, -1, -1):
			if (self._context.bullet_array_ml[i,ARRAY_ML_COLLIDE_MASK] & 1 << self.index):
				self._context.bullet_list[i].vanish()
				self.to_remove = True
		
		
		if keys[INDEX_SHOT]:
			
			foe_aimed_list = []
			for foe in self._context.foe_list:
				if foe.y > self.y and abs(foe.x - self.x) < SHOT_WIDTH / 2:
					foe_aimed_list.append(foe)

			if foe_aimed_list:
				foe_choosen = random.randint(0,len(foe_aimed_list) - 1)
				shot = Shot(self._context)
				shot.x = self.x
				shot.y = self.y
				shot.aimed_foe = foe_aimed_list[foe_choosen]


		return self

	def draw(self):
		if FONKY_LINES:
			gl.glDisable(gl.GL_TEXTURE_2D)
			for i in range(array_fill):
				x,y = bullet_array[:ARRAY_Y+1,i]
				coeff = ((float(self.x)-x)**2+(self.y-y)**2)/LINE_RADIUS2
				if coeff <= 1:
					FONKY_COLOR[3] = (1-coeff) ** 2 # alpha component
					gl.glColor4f(*FONKY_COLOR)
					gl.glBegin(gl.GL_LINES)
					gl.glVertex2f(x, y)
					gl.glVertex2f(self.x, self.y)
					gl.glEnd()
			gl.glColor4f(1.0, 1.0, 1.0, 1.0)

		gl.glPushMatrix()
		gl.glTranslatef(self.x, self.y, 0)
		self.t = (self.t+1)%360
		gl.glRotatef(self.t, 0, 0, 1)
		self.sprite.draw()
		gl.glCallList(self.circle_list)
		gl.glPopMatrix()
		


	def vanish(self):
		if not NO_DEATH:
			self.to_remove = True
			player_list.remove(self)
		else:
			pass
