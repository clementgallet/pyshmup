import Numeric as num
import OpenGL.GL as gl
import OpenGL.GLU as glu

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

	# set by stage
	_field_width = None
	_field_height = None

	# set by event handler
	_screen_width = None
	_screen_height = None

	# computed in _adjust_perspective()
	_view_width = None
	_view_height = None

	old_dump_stats = False

	use_horizontal_scroll = True
	use_vertical_scroll = False

	def __init__(self):
		# lower-level objects are allowed to use thoses structures directly
		self.update_list = []
		self.bullet_list = []
		self.player_list = []
		self.foe_list = []
		self.shot_list = []
		self.others_list = []

		self.array_size = 8
		self.bullet_array = num.zeros((ARRAY_DIM, self.array_size), num.Float)
		self.array_fill = 0

		self.array_ml_size = 8
		self.bullet_array_ml = num.zeros((ARRAY_ML_DIM, self.array_ml_size), num.Float)
		self.array_ml_fill = 0
	
		self.collision = 0
		self.collisionML = 0

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

		if system_state.screen_resized:
			self.set_screen_size(system_state.screen_width, system_state.screen_height)
			system_state.screen_resized = False

		if system_state.dump_stats:
			if not self.old_dump_stats:
				print "stats !"
				pprint([(s, self.__getattribute__(s)) for s in dir(self) if s.find('_list') == -1 and s.find('array') == -1])
			self.old_dump_stats = system_state.dump_stats

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

		camera_x = 0
		camera_y = 0
		if self._scroll_type == SCROLL_HORIZ:
			try:
				px = self.player_list[0].x
				camera_x = px/self._field_width*(self._field_width - self._view_width)
			except IndexError:
				camera_x = 0
		if self._scroll_type == SCROLL_VERT:
			try:
				py = self.player_list[0].y
				camera_y = py/self._field_height*(self._field_height - self._view_height)
			except IndexError:
				camera_y = 0
		gl.glPushMatrix()
		gl.glTranslatef(-camera_x, -camera_y, 0)
		for object in self.player_list + self.foe_list + self.shot_list + self.others_list:
			object.draw()
		#print ("nbr of normal/ml bullets : " + str(self.array_fill) + "/" + str(self.array_ml_fill))
		draw.draw(self.bullet_array, self.array_fill,self.bullet_array_ml, self.array_ml_fill)
#		gl.glAccum(gl.GL_MULT, 0.9)
#		gl.glAccum(gl.GL_ACCUM, 1.0)
#		gl.glAccum(gl.GL_RETURN, 1.0)
		gl.glDisable(gl.GL_TEXTURE_2D)
		gl.glLineWidth(3.)
		gl.glColor4f(1.,1.,1.,1.)
		gl.glBegin(gl.GL_LINE_LOOP)
		gl.glVertex2f(-0.45*self._field_width, 0.45*self._field_height)
		gl.glVertex2f(-0.45*self._field_width,-0.45*self._field_height)
		gl.glVertex2f(0.45*self._field_width, -0.45*self._field_height)
		gl.glVertex2f(0.45*self._field_width,  0.45*self._field_height)
		gl.glEnd()
		gl.glBegin(gl.GL_LINE_LOOP)
		gl.glVertex2f(-0.5*self._field_width, 0.5*self._field_height)
		gl.glVertex2f(-0.5*self._field_width,-0.5*self._field_height)
		gl.glVertex2f(0.5*self._field_width, -0.5*self._field_height)
		gl.glVertex2f(0.5*self._field_width,  0.5*self._field_height)
		gl.glEnd()
		gl.glEnable(gl.GL_TEXTURE_2D)
		gl.glLineWidth(1.)
		

		pygame.display.flip()
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

		gl.glPopMatrix()

	def set_screen_size(self, w, h):
		self._screen_width = w
		self._screen_height = h
		self._adjust_perspective()

	def set_field_size(self, w, h):
		self._field_width = w
		self._field_height = h
		self._adjust_perspective()

		self.left_border = -(w/2.)*(1+OUT_LIMIT)
		self.right_border = (w/2.)*(1+OUT_LIMIT)
		self.down_border = -(h/2.)*(1+OUT_LIMIT)
		self.up_border   =  (h/2.)*(1+OUT_LIMIT)

	def _adjust_perspective(self):
		if self._screen_width is None or self._field_width is None:
			return

		# zoom factors to fit dimensions, units.px^-1
		zoom_x = float(self._field_width) / self._screen_width
		zoom_y = float(self._field_height) / self._screen_height

		# GL viewport, not game view
		#  can change in the lines below
		viewport_width = self._screen_width
		viewport_height = self._screen_height

		if zoom_x > zoom_y:
			if self.use_horizontal_scroll:
				self._scroll_type = SCROLL_HORIZ
				zoom = zoom_y
			else:
				self._scroll_type = SCROLL_NONE
				zoom = zoom_x
				viewport_height = (zoom_y/zoom_x)*self._screen_height
		elif zoom_x < zoom_y:
			if self.use_vertical_scroll:
				self._scroll_type = SCROLL_VERT
				zoom = zoom_x
			else:
				self._scroll_type = SCROLL_NONE
				zoom = zoom_y
				viewport_width = (zoom_x/zoom_y)*self._screen_width
		else:
			self._scroll_type = SCROLL_NONE
			zoom = zoom_x

		# view size (in game units)
		self._view_width = zoom*viewport_width
		self._view_height = zoom*viewport_height

		gl.glViewport( int((self._screen_width-viewport_width)/2),
		               int((self._screen_height-viewport_height)/2),
		               int(viewport_width),
		               int(viewport_height))

		# setting perspective
		gl.glMatrixMode(gl.GL_PROJECTION)
		gl.glLoadIdentity()
		fov = 30
		dist = self._view_height / (2 * math.tan(fov*math.pi/360))
		glu.gluPerspective(fov, float(self._view_width)/self._view_height, dist*0.5, dist*1.5)
		gl.glMatrixMode(gl.GL_MODELVIEW)
		gl.glLoadIdentity()
		gl.glTranslate(0, 0, -dist)

		



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
			time_x = NEVER
		elif 0 < direction % 360 < 180:
			time_x = (self.right_border - x) / \
			             (math.sin(direction*math.pi/180)*speed)
		else:
			time_x = (self.left_border - x) / \
			             (math.sin(direction*math.pi/180)*speed)
	
		if abs(direction%180 - 90) < 0.1 or speed == 0:
			time_y = NEVER
		elif 90 < direction % 360 < 270:
			time_y = (self.up_border - y) / \
			             (-math.cos(direction*math.pi/180)*speed)
		else:
			time_y = (self.down_border - y) / \
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
		self.bullet_array[ARRAY_OUT_TIME] -= 1.
		if not (self.frame % 10):
			for i in range(self.array_fill - 1, -1, -1):
				#print self.bullet_array[ARRAY_OUT_TIME]
				if self.bullet_array[ARRAY_OUT_TIME,i] < 0:
					self.delete_bullet(i)
					#print("delete bullet : out bounds")


	def _check_collisions(self):
		self.bullet_array[ARRAY_UNTIL] -= 1.
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
