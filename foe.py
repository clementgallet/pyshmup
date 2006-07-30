
import math

from constants import *

from OpenGL.GL import *

from bulletml import BulletMLController
import sprite
from bullet import SimpleBulletML


class Foe(object):
	def __init__(self):
		#FIXME: remove these defaults
		self.x = UNIT_WIDTH*0
		self.y = UNIT_HEIGHT*0.3
		self.z = 0
		self.direction = 0
		self.speed = 0
		self.to_remove = False
		self.sprite = sprite.get_sprite( FOE_BITMAP )
		self.bullet_sprite = sprite.get_sprite( BULLET_BITMAP )
		self.life = FOE_LIFE
		self.frame = 0

	def spawn(self,context):
		self._context = context
		context.update_list.append(self)
		context.foe_list.append(self)


	def draw(self):
		glPushMatrix()
		#TODO: if foe has no target, it becomes transparent
		glColor4f(1.0, 1.0, 1.0, 1.0)
		glTranslatef(self.x, self.y, 0)
		self.sprite.draw()
		glPopMatrix()


	def update(self):
		self.direction = self.direction % 360
		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		   self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT) or self.life < 0:
			self.vanish()
			return self
		self.x += math.sin(self.direction*math.pi/180)*self.speed
		self.y -= math.cos(self.direction*math.pi/180)*self.speed

		for shot in self._context.shot_list:
			if max(abs(shot.x - self.x),abs(shot.y - self.y)) < FOE_RADIUS:
				self.life -= 1
				shot.vanish()
		
		return self

	def fireml(self, direction=None, speed=None, new_bullet=None):
		if direction is not None:
			new_bullet.direction = direction
		else:
			new_bullet.direction = self.direction
		if speed is not None:
			new_bullet.speed = speed
		else:
			new_bullet.speed = self.speed
			
		self._context.bullet_array[:ARRAY_Z+1,new_bullet.index] = [self.x,self.y,self.z]
		new_bullet.sprite = self.bullet_sprite
		self._context.bullet_array[ARRAY_LIST,new_bullet.index] = new_bullet.sprite.list


#	def firenoml(self, direction=None, speed=None, new_bullet=None):
#		self.z -= 0.0001
#		if direction is None:
#			direction = self.direction
#		if speed is None:
#			speed = self.speed
#			
#
#		bullet_array[:ARRAY_Z+1,new_bullet.index] = [self.x,self.y,self.z+0.00001]
#		new_bullet.sprite = self.bullet_sprite
#		bullet_array[ARRAY_LIST][new_bullet.index] = new_bullet.sprite.list



	def vanish(self):
		self._context.foe_list.remove(self)
		self.to_remove = True

class BulletMLFoe(Foe):
	#FIXME: refactor this and SimpleBulletML
	def __init__(self, bulletml_behav):

		self.controller = BulletMLController()
		self.controller.game_object = self
		self.controller.set_behavior(bulletml_behav)
		self.wait = 0
		super(BulletMLFoe, self).__init__()
	
	def spawn(self,context):
		#FIXME: the first player is always aimed, ahahah !!
		if context.player_list:
			self.aimed_player = context.player_list[0]
		else:
			self.aimed_player = None
		super(BulletMLFoe, self).spawn(context)
		

	def fireml(self, controller, direction=None, speed=None):
		new_bullet = SimpleBulletML(self._context)
		new_bullet.controller = controller
		controller.set_game_object(new_bullet)
		new_bullet.aimed_player = self.aimed_player
		super(BulletMLFoe, self).fireml(direction, speed, new_bullet)


	def firenoml(self, direction=None, speed=None):
		if direction is None:
			direction = self.direction
		if speed is None:
			speed = self.speed
		
		self._context.create_bullet(self.x, self.y, self.z, direction, speed, self.bullet_sprite.list)


	def update(self):
		if self.wait > 0:
			self.wait -= 1
		else:
			self.controller.run()

		return super(BulletMLFoe, self).update()
