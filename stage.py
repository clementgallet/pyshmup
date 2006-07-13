import logging
import copy
import xml.dom.minidom

from constants import *

from foe import BulletMLFoe
import sprite

l = logging.getLogger('stage')
l.setLevel(logging.DEBUG)

class StageLoader(object):
	"""
	Stores information about a stage, and can spawn objects in time.

	There is no way to cleanly "remove" a stage.
	"""

	to_remove = False
	
	def __init__(self, context, stage_file=None):
		self._context = context
		if stage_file is not None:
			self.load(stage_file)
		self.frame = 0
		self.launch_list = []

	def update(self):
		"""
		Advance the stage by a frame.

		This function populates the game context as needed (ie. spawns
		objects as they are meant to come).
		"""
		while (self.launch_list and self.launch_list[0].frame <= self._context.frame):
			obj = self.launch_list.pop(0)
		return self

	def load(self, stage_file):
		"""
		Load the stage contents from a stage file.
		"""
		print [stage_file]
		doc =  xml.dom.minidom.parse(stage_file)
		root = doc.documentElement

		launch_list = []

		for foe_node in root.getElementsByTagName("foe"):
			behav_name = foe_node.getElementsByTagName('behav')[0].childNodes[0].nodeValue
			bullet_name = foe_node.getElementsByTagName('bullet')[0].childNodes[0].nodeValue
			sprite_name = foe_node.getElementsByTagName('sprite')[0].childNodes[0].nodeValue
			x = eval(foe_node.getElementsByTagName('x')[0].childNodes[0].nodeValue)
			y = eval(foe_node.getElementsByTagName('y')[0].childNodes[0].nodeValue)
			frame = eval(foe_node.getElementsByTagName('frame')[0].childNodes[0].nodeValue)
			
			foe = BulletMLFoe(self._context, BEHAV_PATH + behav_name)
			foe.x = x
			foe.y = y
			foe.sprite = sprite.get_sprite(BITMAP_PATH + sprite_name)
			foe.bullet_sprite = sprite.get_sprite(BITMAP_PATH + bullet_name)
			launch_list.append(foe)

			
		self.launch_list = launch_list
		self._stage_file = stage_file

