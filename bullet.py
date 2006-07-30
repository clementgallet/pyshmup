from constants import *

from bulletml import BulletMLController
import sprite

from pprint import pprint
	

class SimpleBulletML(object):

	to_remove = False

	def __init__(self, context, bulletml_behav=None):
	
		global bullet_list_length
		
		self._context = context
		
		context.update_list.append(self)
		context.bullet_list.append(self)
		context.bullet_list_length += 1
		
		self.controller = BulletMLController()
		self.controller.game_object = self
		if bulletml_behav is not None:
			self.controller.set_behavior(bulletml_behav)
		
		self.wait = 0

		self.sprite = sprite.get_sprite(BULLET_BITMAP)

		self.index = context.create_bullet(0, 0, 0, 0, 0, self.sprite.list)
		
		context.bullet_array[ARRAY_LIST_INDEX][self.index] = context.bullet_list_length - 1
		context.bullet_array[ARRAY_STATE][self.index] = ARRAY_STATE_ML
	
	def getx(self):
		return self._context.bullet_array[ARRAY_X,self.index]
	x = property(getx)
	
	def gety(self):
		return self._context.bullet_array[ARRAY_Y,self.index]
	y = property(gety)
	
	def getz(self):
		return self._context.bullet_array[ARRAY_Z,self.index]
	z = property(getz)
	
	def getd(self):
		return self._context.bullet_array[ARRAY_DIRECTION][self.index]
	def setd(self,d):
		self._context.bullet_array[ARRAY_DIRECTION][self.index] = d
	direction = property(getd,setd)
	
	def gets(self):
		return self._context.bullet_array[ARRAY_SPEED][self.index]
	def sets(self,s):
		self._context.bullet_array[ARRAY_SPEED][self.index] = s	
	speed = property(gets,sets)
		

	def vanish(self):
		self.to_remove = True

		self._context.delete_bullet(self.index)


	def fire_complex(self, controller, direction=None, speed=None):
		new_bullet = SimpleBulletML()
		new_bullet.controller = controller
		controller.set_game_object(new_bullet)
		new_bullet.aimed_player = self.aimed_player

		if direction is not None:
			new_bullet.direction = direction
		else:
			new_bullet.direction = self.direction
		if speed is not None:
			new_bullet.speed = speed
		else:
			new_bullet.speed = self.speed
			
		bullet_array[:ARRAY_Z+1,new_bullet.index] = self.x,self.y,self.z+0.0001
		new_bullet.sprite = self.sprite
		bullet_array[ARRAY_LIST][new_bullet.index] = new_bullet.sprite.list

	def fire(self, direction=None, speed=None):
		if direction is None:
			direction = self.direction
		if speed is None:
			speed = self.speed
		
		self._context.create_bullet(self.x, self.y, self.z, direction, speed, self.sprite.list)

	def update(self):
		if self.wait > 0:
			self.wait -= 1
		else:
			self.controller.run()

		if self.to_remove:
			return self

		try:
			if self._context.bullet_array[ARRAY_LIST_INDEX, self.index] != self._context.bullet_list.index(self):
				print "error !"
				print "in array:", self._context.bullet_array[ARRAY_LIST_INDEX, self.index]
				print "in reality:",  self._context.bullet_list.index(self)
		except ValueError:
			pprint(self._context.bullet_list)
			pprint(self)
			pprint(self._context.update_list)
			print "\n---------\n\n"
			#pass

		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		       self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT):
			self.vanish()

		return self	
