import Numeric as num
import OpenGL.GL as gl

from constants import *
import math
import pygame

import player
import stage

import draw

class GameContext(object):
	"""
	Stores, updates, and draws a game state.
	"""
	
	# End of game marker
	done = False

	frame = 0

	def __init__(self):
		# lower-level objects are allowed to use thoses structures directly
		self.update_list = []
		self.bullet_list = []
		self.bullet_list_length = 0
		self.player_list = []
		self.foe_list = []
		self.shot_list = []

		self.array_size = 8
		self.bullet_array = num.zeros((ARRAY_DIM, self.array_size), num.Float)
		self.array_fill = 0

		player.Player(self)

	####################################
	#  Interface for higher-level usage

	def load_stage(self, stage_name):
		"""
		Create an initial game state from a stage file.
		"""
		self.__init__() # reinit
		self.update_list.append(stage.StageLoader(self, stage_name))
	
	def update(self, system_state):
		"""
		Advance the game state by a frame.
		"""
		#REMOVEME
		#print "updating"
		#return

		# FIXME: move this in a game manager or something (that can do pause and such...)
		if system_state.keys[pygame.K_ESCAPE]:
			self.done = True

		# Share current system state with other objects
		self._system_state = system_state

		# Update everything
		self._move_bullets()
		self._update_objects()

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
		draw.draw(self.bullet_array, self.array_fill)
#		glAccum(GL_MULT, 0.9)
#		glAccum(GL_ACCUM, 1.0)
#		glAccum(GL_RETURN, 1.0)
		
		pygame.display.flip()
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

	#######################################
	# Services to lower-level game objects

	def create_bullet(self, x, y, z, direction, speed, display_list):
		"""
		Book a slot in the big bullet array, and return its index.
		"""
		index = self.array_fill
		self.array_fill += 1

		# Grow array
		if self.array_fill == self.array_size:
			new_array = num.zeros((ARRAY_DIM,2*self.array_size),num.Float)
			new_array[:,:self.array_size] = self.bullet_array
			self.bullet_array = new_array
			self.array_size *= 2
		
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
	
#		bullet_array[ARRAY_X][index] = x
#		bullet_array[ARRAY_Y][index] = y
#		bullet_array[ARRAY_Z][index] = z
#		bullet_array[ARRAY_DIRECTION][index] = direction
#		bullet_array[ARRAY_SPEED][index] = speed
#		bullet_array[ARRAY_LIST][index] = display_list
#		bullet_array[ARRAY_STATE][index] = ARRAY_STATE_DANG
#		bullet_array[ARRAY_UNTIL][index] = 0
#		bullet_array[ARRAY_LIST][index] = -1
#		bullet_array[ARRAY_OUT_TIME][index] = out_time
		self.bullet_array[:,index] = (x,y,z,direction,speed,display_list,ARRAY_STATE_DANG,0,-1,out_time)
	
		return index
	
	def delete_bullet(self, index):
		"""
		Free a slot in the big bullet array.
		"""
		# Decrease size and fill emptied slot
		self.array_fill -= 1
		if self.array_fill != index:
			self.bullet_array[:, index] = self.bullet_array[:, self.array_fill]

		# Bullet list index stored in array
		list_index = int(self.bullet_array[ARRAY_LIST_INDEX, index])
		if list_index >= 0:
			# This is a complex bullet
			bullet_list[list_index].index = index


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

	def _update_objects(self):
		self.update_list = [obj for obj in self.update_list if obj.update().to_remove == False]
	
