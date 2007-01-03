from constants import *

from bulletml import BulletMLController
import sprite

class SimpleBulletML(object):

	to_remove = False

	def __init__(self, context):
	
		self._context = context
		
		context.update_list.append(self)
		self.index = len(self._context.bullet_list)
		context.bullet_list.append(self)
		
		self.controller = BulletMLController()
		self.controller.game_object = self
		
		self.wait = 0

		self.sprite = sprite.get_sprite(BULLET_BITMAP)
	
	def getx(self):
		return self._context.bullet_array_ml[ARRAY_ML_X,self.index]
	x = property(getx)
	
	def gety(self):
		return self._context.bullet_array_ml[ARRAY_ML_Y,self.index]
	y = property(gety)

	def getz(self):
		return self._context.bullet_array_ml[ARRAY_ML_Z,self.index]
	z = property(getz)
	
	def getd(self):
		return self._context.bullet_array_ml[ARRAY_ML_DIRECTION][self.index]

	def setd(self,d):
		self._context.bullet_array_ml[ARRAY_ML_DIRECTION][self.index] = d
	direction = property(getd,setd)
	
	def gets(self):
		return self._context.bullet_array_ml[ARRAY_ML_SPEED][self.index]
	def sets(self,s):
		self._context.bullet_array_ml[ARRAY_ML_SPEED][self.index] = s	
	speed = property(gets,sets)


	def vanish(self):
		self.to_remove = True
		self._context.delete_bullet_ml(self.index)

	def fire_complex(self, controller, direction=None, speed=None):
		new_bullet = SimpleBulletML(self._context)
		new_bullet.controller = controller
		controller.set_game_object(new_bullet)
		new_bullet.aimed_player = self.aimed_player
		new_bullet.sprite = self.sprite

		if direction is None:
			direction = self.direction
		if speed is None:
			speed = self.speed
			
		self._context.create_bullet_ml(self.x, self.y, self.z+0.001, direction, speed, self.sprite.list)

	def fire(self, direction=None, speed=None):
		if direction is None:
			direction = self.direction
		if speed is None:
			speed = self.speed
		
		self._context.create_bullet(self.x, self.y, self.z+0.001, direction, speed, self.sprite.list)

	def update(self):
		if self.to_remove:
			return self
		
		if self.wait > 0:
			self.wait -= 1
		else:
			self.controller.run()

		if self.to_remove:
			return self

		if self.x < self._context.left_border  or self.x > self._context.right_border or \
		       self.y < self._context.down_border or self.y > self._context.up_border:
			self.vanish()

		return self
