from constants import *

from bulletml import BulletMLController
import sprite

class SimpleBulletML(object):

	to_remove = False

	def __init__(self, context):
	
		self._context = context
		self.x = 0
		self.y = 0
		self.direction = 0
		self.speed = 0
		
		context.update_list.append(self)
		self.index = len(context.bullet_list)
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
		self._context.bullet_list.remove(self)
		delete_bullet_ml(self.index)

	def fire_complex(self, controller, direction=None, speed=None):
		new_bullet = SimpleBulletML(self._context)
		new_bullet.controller = controller
		controller.set_game_object(new_bullet)
		new_bullet.aimed_player = self.aimed_player
		new_bullet.sprite = self.sprite

		if direction is not None:
			new_bullet.direction = direction
		else:
			new_bullet.direction = self.direction
		if speed is not None:
			new_bullet.speed = speed
		else:
			new_bullet.speed = self.speed
			
		self._context.create_bullet_ml(self.x, self.y, self.z+0.001, new_bullet.direction, new_bullet.speed, self.sprite.list)

	def fire(self, direction=None, speed=None):
		if direction is None:
			direction = self.direction
		if speed is None:
			speed = self.speed
		
		self._context.create_bullet(self.x, self.y, self.z+0.001, direction, speed, self.sprite.list)

	def update(self):
		if self.wait > 0:
			self.wait -= 1
		else:
			self.controller.run()

		if self.to_remove:
			return self

		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		       self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT):
			self.vanish()

		return self
