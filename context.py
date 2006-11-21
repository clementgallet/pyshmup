import Numeric as num
import OpenGL.GL as gl

from constants import *
import math
import pygame

import player
import stage
import sprite

import draw
import coll

from pprint import pprint

import logging
l = logging.getLogger('context')
l.setLevel(logging.DEBUG)

class GameContext(object):
	"""
	Stores, updates, and draws a game state.
	"""
	
	frame = 0

	def __init__(self):
		# lower-level objects are allowed to use thoses structures directly
		self.update_list = []
		self.bullet_list = []
		self.player_list = []
		self.foe_list = []
		self.shot_list = []

		self.array_size = 8
		self.bullet_array = num.zeros((ARRAY_DIM, self.array_size), num.Float)
		self.array_fill = 0

		self.array_ml_size = 8
		self.bullet_array_ml = num.zeros((ARRAY_ML_DIM, self.array_ml_size), num.Float)
		self.array_ml_fill = 0
	
		self.collision = 0
		self.collisionML = 0
		player.Player(self)

		#self.tr = sprite.get_sprite(BULLET_BITMAP)

	####################################
	#  Interface for higher-level usage

	def load_stage(self):
		"""
		Create an initial game state from a stage file.
		"""
		self.__init__() # reinit
		self.update_list.append(stage.Stage1(self))
	
	def update(self, system_state):
		"""
		Advance the game state by a frame.
		"""
		#REMOVEME
		#print "updating"
		#return

		# Share current system state with other objects
		self._system_state = system_state

		# Update everything
		self._move_bullets()
		self._out_bounds()
		self._check_collisions()
		self._update_objects()

		#self._check_array()

		self.frame += 1

	def draw(self):
		"""
		Draw the game state.
		"""
		#REMOVEME
		#print "drawing"
		#return

		for object in self.player_list + self.foe_list + self.shot_list:
			object.draw()
		#print ("nbr of normal/ml bullets : " + str(self.array_fill) + "/" + str(self.array_ml_fill))
		draw.draw(self.bullet_array, self.array_fill,self.bullet_array_ml, self.array_ml_fill)
#		gl.glAccum(gl.GL_MULT, 0.9)
#		gl.glAccum(gl.GL_ACCUM, 1.0)
#		gl.glAccum(gl.GL_RETURN, 1.0)
		
		pygame.display.flip()
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

	#######################################
	# Services to lower-level game objects

	def create_bullet(self, x, y, z, direction, speed, display_list):
		"""
		Book a slot in the big bullet array, and return its index.
		"""
		# Calculate time before the bullet is out of screen
		# (the +10 / -10 on the first line
		# is to avoid wrapping of the modulo around 0)
		if abs(((direction + 10)%180)-10) < 0.1 or speed == 0:
			time_x = 10000000
		elif 0 < direction % 360 < 180:
			time_x = (UNIT_WIDTH*(1+OUT_LIMIT) - x) / \
			             (math.sin(direction*math.pi/180)*speed)
		else:
			time_x = (-UNIT_WIDTH*(1+OUT_LIMIT) - x) / \
			             (math.sin(direction*math.pi/180)*speed)
	
		if abs(direction%180 - 90) < 0.1 or speed == 0:
			time_y = 10000000
		elif 90 < direction % 360 < 270:
			time_y = (UNIT_HEIGHT*(1+OUT_LIMIT) - y) / \
			             (-math.cos(direction*math.pi/180)*speed)
		else:
			time_y = (-UNIT_HEIGHT*(1+OUT_LIMIT) - y) / \
			             (-math.cos(direction*math.pi/180)*speed)
	
		out_time = min(time_x,time_y)
		#print ("out_time : " + str(out_time))	
#		bullet_array[ARRAY_X][index] = x
#		bullet_array[ARRAY_Y][index] = y
#		bullet_array[ARRAY_Z][index] = z
#		bullet_array[ARRAY_DIRECTION][index] = direction
#		bullet_array[ARRAY_SPEED][index] = speed
#		bullet_array[ARRAY_LIST][index] = display_list
#		bullet_array[ARRAY_UNTIL][index] = 0
#		bullet_array[ARRAY_OUT_TIME][index] = out_time
#		bullet_array[ARRAY_COLLIDE_MASK][index] = 0
		#print("x : " + str(x) +  ", y : " + str(y))
		self.bullet_array[:,self.array_fill] = (x,y,z,direction,speed,display_list,0,out_time,0)
	
		self.array_fill += 1

		# Grow array
		if self.array_fill == self.array_size:
			new_array = num.zeros((ARRAY_DIM,2*self.array_size),num.Float)
			new_array[:,:self.array_size] = self.bullet_array
			self.bullet_array = new_array
			self.array_size *= 2
		
	def create_bullet_ml(self, x, y, z, direction, speed, display_list):
		"""
		Book a slot in the big bullet array, and return its index.
		"""
#		bullet_array_ml[ARRAY_ML_X][index] = x
#		bullet_array_ml[ARRAY_ML_Y][index] = y
#		bullet_array_ml[ARRAY_ML_Z][index] = z
#		bullet_array_ml[ARRAY_ML_DIRECTION][index] = direction
#		bullet_array_ml[ARRAY_ML_SPEED][index] = speed
#		bullet_array_ml[ARRAY_ML_LIST][index] = display_list
#		bullet_array_ml[ARRAY_ML_COLLIDE_MASK][index] = 0
		self.bullet_array_ml[:,self.array_ml_fill] = (x,y,z,direction,speed,display_list,0)
		self.array_ml_fill += 1
#		print("add bullet : #" + str(self.array_ml_fill))

		# Grow array
		if self.array_ml_fill == self.array_ml_size:
			new_array = num.zeros((ARRAY_ML_DIM,2*self.array_ml_size),num.Float)
			new_array[:,:self.array_ml_size] = self.bullet_array_ml
			self.bullet_array_ml = new_array
			self.array_ml_size *= 2
		
	
	def delete_bullet(self, index):
		"""
		Free a slot in the big bullet array.
		"""
		
		# Decrease size and fill emptied slot
		self.array_fill -= 1
		if self.array_fill != index:
			self.bullet_array[:, index] = self.bullet_array[:, self.array_fill]

	def delete_bullet_ml(self, index):
		"""
		Free a slot in the big bullet array.
		"""
		if index >= self.array_ml_fill:
			print "array : " + str(self.array_ml_fill) + " - index : " + str(index)
		# Decrease size and fill emptied slot
		self.array_ml_fill -= 1
#		print("delete bullet "+str(index)+" : #" + str(self.array_ml_fill))
		bullet = self.bullet_list.pop()
		if index > self.array_ml_fill:
			print "index sup as array_fill"
		if self.array_ml_fill != index:
			self.bullet_array_ml[:, index] = self.bullet_array_ml[:, self.array_ml_fill]
			bullet.index = index
			self.bullet_list[index] = bullet
		

	###################
	# Internal methods

	def _move_bullets(self):
		num.add( \
		  self.bullet_array[ARRAY_X], \
		  num.multiply( \
		    num.sin( \
		      num.multiply( \
		        self.bullet_array[ARRAY_DIRECTION], \
		        math.pi/180)), \
		    self.bullet_array[ARRAY_SPEED]), \
		  self.bullet_array[ARRAY_X])
		
		num.subtract( \
		  self.bullet_array[ARRAY_Y], \
		  num.multiply( \
		    num.cos( \
		      num.multiply( \
		        self.bullet_array[ARRAY_DIRECTION], \
		        math.pi/180)), \
		    self.bullet_array[ARRAY_SPEED]), \
		  self.bullet_array[ARRAY_Y]) 

		num.add( \
		  self.bullet_array_ml[ARRAY_ML_X], \
		  num.multiply( \
		    num.sin( \
		      num.multiply( \
		        self.bullet_array_ml[ARRAY_ML_DIRECTION], \
		        math.pi/180)), \
		    self.bullet_array_ml[ARRAY_ML_SPEED]), \
		  self.bullet_array_ml[ARRAY_ML_X])
		
		num.subtract( \
		  self.bullet_array_ml[ARRAY_ML_Y], \
		  num.multiply( \
		    num.cos( \
		      num.multiply( \
		        self.bullet_array_ml[ARRAY_ML_DIRECTION], \
		        math.pi/180)), \
		    self.bullet_array_ml[ARRAY_ML_SPEED]), \
		  self.bullet_array_ml[ARRAY_ML_Y]) 

	def _out_bounds(self):
		num.subtract(self.bullet_array[ARRAY_OUT_TIME],num.ones((self.array_size),num.Float),self.bullet_array[ARRAY_OUT_TIME])
		for i in range(self.array_fill - 1, -1, -1):
			#print self.bullet_array[ARRAY_OUT_TIME]
			if self.bullet_array[ARRAY_OUT_TIME,i] < 0:
				self.delete_bullet(i)
				#print("delete bullet : out bounds")


	def _check_collisions(self):
		num.subtract(self.bullet_array[ARRAY_UNTIL],num.ones((self.array_size),num.Float),self.bullet_array[ARRAY_UNTIL])
		self.collision = coll.coll(self.bullet_array, self.array_fill, self.player_list, len(self.player_list))
		self.collisionML = coll.collml(self.bullet_array_ml, self.array_ml_fill, self.player_list, len(self.player_list))

	def _update_objects(self):
		self.update_list = [obj for obj in self.update_list if obj.update().to_remove == False]
	

	## DEBUG

	def _check_array(self):
		output = []
		need_output = False
		for i in range(self.array_fill):
			if self.bullet_array[ARRAY_LIST_INDEX, i] != -1:
				index = int(self.bullet_array[ARRAY_LIST_INDEX, i])
				try:
					bullet = self.bullet_list[index]
				except IndexError:
					need_output = True
					output.append("LIST_INDEX out of range : it is : %i, length is %i" % (index, len(self.bullet_list)))
		if need_output:
			l.error("\n\n\n-----\n\nInvalid array : LIST_INDEX out of range (at frame %i)" % self.frame)
			for message in output:
				l.error(message)
			self._dump_array()
					
	def _dump_array(self):
		l.debug('Soon : the array.')
		for i in xrange(self.array_fill):
			l.debug('Array elem %i' % i)
			l.debug('  X : %f' % self.bullet_array[ARRAY_X, i])
			l.debug('  Y : %f' % self.bullet_array[ARRAY_Y, i])
			l.debug('  Z : %f' % self.bullet_array[ARRAY_Z, i])
			l.debug('  DIRECTION : %f' % self.bullet_array[ARRAY_DIRECTION, i])
			l.debug('  SPEED : %f' % self.bullet_array[ARRAY_SPEED, i])
			l.debug('  LIST : %f' % self.bullet_array[ARRAY_LIST, i])
			l.debug('  UNTIL : %f' % self.bullet_array[ARRAY_UNTIL, i])
			l.debug('  LIST_INDEX : %f' % self.bullet_array[ARRAY_LIST_INDEX, i])
			l.debug('  OUT_TIME : %f' % self.bullet_array[ARRAY_OUT_TIME, i])
