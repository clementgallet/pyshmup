import logging
import xml.dom.minidom

from constants import *

from foe import BulletMLFoe
import sprite
import text

l = logging.getLogger('stage')
l.setLevel(logging.DEBUG)

class StageLoader(object):
	"""
	Stores information about a stage, and can spawn objects in time.

	There is no way to cleanly "remove" a stage.
	"""

	to_remove = False
	
	def __init__(self, context):
		self._context = context
		self.launch_list = []

		# default values
		context.set_field_size(DEFAULT_FIELD_WIDTH, DEFAULT_FIELD_HEIGHT)

	def update(self):
		"""
		Advance the stage by a frame.

		This function populates the game context as needed (ie. spawns
		objects as they are meant to come).
		"""

		while (self.launch_list and self.launch_list[0].frame <= self._context.frame):
			obj = self.launch_list.pop(0)
			obj.spawn(self._context)
		return self

	def create(self, ml, x, y, frame, bullet, sprite_name):
		foe = BulletMLFoe(BEHAV_PATH + ml)
		foe.x = x
		foe.y = y
		foe.sprite = sprite.get_sprite(BITMAP_PATH + sprite_name)
		foe.bullet_sprite = sprite.get_sprite(BITMAP_PATH + bullet)
		foe.frame = frame
		self.launch_list.append(foe)

import fonts

class Stage1(StageLoader):
	
	def __init__(self, context):
		super(Stage1, self).__init__(context)
		context.set_field_size(200, 400)
		self.create('th.xml', -50, 100, 0, 'shot.png', 'foe.png')
		self.create('th.xml', 50, 100, 50, 'shot.png', 'foe.png')
		self.create('th.xml', 0, 100, 100, 'shot.png', 'foe.png')
		self.create('th.xml', 0, 150, 150, 'shot.png', 'foe.png')
		self.launch_list.append(text.StaticText('foobar', 0, 0, 20))
