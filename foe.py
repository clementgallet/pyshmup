
import math

from constants import *

import OpenGL.GL as gl

from bulletml import BulletMLController
import sprite
from bullet import SimpleBulletML


class Foe(object):
	x = 0
	y = 0
	z = 0
	speed = 0
	direction = 0

	def __init__(self):
		#FIXME: remove these defaults
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
		gl.glPushMatrix()
		#TODO: if foe has no target, it becomes transparent
		gl.glColor4f(1.0, 1.0, 1.0, 1.0)
		gl.glTranslatef(self.x, self.y, 0)
		self.sprite.draw()
		gl.glPopMatrix()


	def update(self):
		self.direction = self.direction % 360
		if self.x < self._context.left_border  or self.x > self._context.right_border or \
		   self.y < self._context.down_border or self.y > self._context.up_border or self.life < 0:
			self.vanish()
			return self
		self.x += math.sin(self.direction*math.pi/180)*self.speed
		self.y -= math.cos(self.direction*math.pi/180)*self.speed

		for shot in self._context.shot_list:
			if max(abs(shot.x - self.x),abs(shot.y - self.y)) < FOE_RADIUS:
				self.life -= 1
				shot.vanish()
		
		return self

	def fire_complex(self, controller, direction=None, speed=None):
		
		new_bullet = SimpleBulletML(self._context)
		new_bullet.controller = controller
		controller.set_game_object(new_bullet)
		new_bullet.aimed_player = self.aimed_player
		new_bullet.sprite = self.bullet_sprite
		
		if direction is None:
			direction = self.direction
		if speed is None:
			speed = self.speed
		self._context.create_bullet_ml(self.x, self.y, self.z+0.001, direction, speed, self.bullet_sprite.list)


	def fire(self, direction=None, speed=None):
		if direction is None:
			direction = self.direction
		if speed is None:
			speed = self.speed
		
		self._context.create_bullet(self.x, self.y, self.z, direction, speed, self.bullet_sprite.list)

	def vanish(self):
		self._context.foe_list.remove(self)
		self.to_remove = True

class BulletMLFoe(Foe):
	#FIXME: refactor this and SimpleBulletML
	def __init__(self, bulletml_behav):
		super(BulletMLFoe, self).__init__()
		self.controller = BulletMLController()
		self.controller.game_object = self
		self.controller.set_behavior(bulletml_behav)
		self.wait = 0
	
	def spawn(self,context):
		super(BulletMLFoe, self).spawn(context)
		#FIXME: the first player is always aimed, ahahah !!
		if context.player_list:
			self.aimed_player = context.player_list[0]
		else:
			self.aimed_player = None

	def update(self):
		if self.wait > 0:
			self.wait -= 1
		else:
			self.controller.run()

		return super(BulletMLFoe, self).update()
