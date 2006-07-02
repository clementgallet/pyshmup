import logging
import copy

from constants import *
from globals import *

from foe import BulletMLFoe
import sprite

l = logging.getLogger('stage')
l.setLevel( logging.DEBUG )

class FoeInfo(object):

	behav = None
	x = 0
	y = 0
	frame = 0
	bullet = None
	sprite = None

class Stage(object):
	def __init__(self, stage_file):
		self.foe_list = copy.deepcopy(StagetoFoeList(stage_file).getFoes())
		self.frame = 0

	def update(self):
		while (self.foe_list and self.foe_list[0].frame == self.frame):
			foe = self.foe_list.pop(0)
			self.launch(foe.behav,foe.x,foe.y,foe.sprite,foe.bullet)
		self.frame += 1

	def launch(self,foe_controller,x,y,foe_bit,bullet_bit):
		foe = BulletMLFoe(BEHAV_PATH + foe_controller)
		foe.x = x
		foe.y = y
		if foe_bit is not None:
			foe.sprite = sprite.get_sprite(BITMAP_PATH + foe_bit)
		else:
			foe.sprite = sprite.get_sprite(FOE_BITMAP)
		if bullet_bit is not None:
			foe.bullet_sprite = sprite.get_sprite(BITMAP_PATH + bullet_bit)
		else:
			foe.bullet_sprite = sprite.get_sprite(BULLET_BITMAP)

class StagetoFoeList(object):

	root = None
	foe_list = []

	def __init__(self,FILE):
		self.file = FILE
		from xml.dom.minidom import parse
		self.doc = parse(self.file)
		self.root = self.doc.documentElement

	def getFoes(self):

		#if self.foe_list is not None:
		#	return self.foe_list

		#self.foe_list = []

		for foe in self.root.getElementsByTagName("foe"):

			#if foe.nodeType == foe.ELEMENT_NODE:
				
			f = FoeInfo()

			for xml in ['behav','bullet','sprite']:

				try:
					f.__setattr__(xml,foe.getElementsByTagName(xml)[0].childNodes[0].nodeValue)
				except:
					l.warning(xml + ' missing')

			for value in ['x','y','frame']:

				try:
					f.__setattr__(value,eval(foe.getElementsByTagName(value)[0].childNodes[0].nodeValue))
				except:
					l.warning(value + ' missing')

			self.foe_list.append(f)
			
		return self.foe_list

