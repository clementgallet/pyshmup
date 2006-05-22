# -*- coding: utf-8 -*-

import logging
import re
import xml.sax, xml.sax.handler
import copy

import traceback

############
## Logging

if __name__ == '__main__':
	console = logging.StreamHandler( )
	console.setLevel( logging.DEBUG )
	formatter = logging.Formatter( '%(name)-12s: %(levelname)-8s %(message)s' )
	console.setFormatter( formatter )
	logging.getLogger('').addHandler( console )

l = logging.getLogger('bulletml')
l.setLevel( logging.DEBUG )

LAUNCHED = 1

class Stage:
	def __init__(self):
		global namespace
		namespace = self
		self.foe_list = []

	def run(self,game_object_control):
		#We suppose that the foe are sorted in the list
		if self.foe_list:
			while self.foe_list and self.foe_list[0].run(game_object_control) == LAUNCHED:
				self.foe_list.remove(self.foe_list[0])


class Foe:
	def __init__(self):
		self.x = 0
		self.y = 0
		self.frame = 1
		
	def run(self,game_object_control):
		if self.frame == game_object_control.game_object.frame:
			try:
				sprite = self.sprite
			except:
				sprite = None

			try:
				bullet = self.bullet
			except:
				bullet = None

			game_object_control.game_object.launch(self.behav,self.x,self.y,sprite,bullet)
			return LAUNCHED
		else:
			return 0

##Builders

target_classes = { 
	'stage'   : Stage,
	'foe'     : Foe,
	}



def get_eval(text):
	try:
		return eval(text)
	except:
		l.warning(text + " is not a number in element")
		return 0

class Builder(object):
	def __init__(self):
		try:
			self.target = target_classes[self.element_name]()
		except KeyError:
			pass # do nothing for classes that don't build a .target


	def add_to( self, builder ):
		# $visitor calls $child.add_to_$($visitor.TYPE), passing himself as the argument on which to operate
		try:
			add_method=self.__getattribute__('add_to_' + builder.element_name )
		except:
			l.error( "Don't know what to do with %s in %s." % (self.element_name, builder.element_name) )
			return
		add_method( builder )
		

	def add_text( self, text ):
		if text:
			l.debug( "Ignoring text : " + text + " in " + self.element_name + "." )


class TextBuilder(Builder):
	text = ''
	def add_text(self, text):
		# Quadratic, but it should not matter, really.
		self.text += text


class StageBuilder(Builder):
	element_name = "stage"


class FoeBuilder(Builder):
	element_name="foe"

	def add_to_stage(self,stage_builder):
		stage_builder.target.foe_list.append(self.target)

class BehavBuilder(TextBuilder):
	element_name="behav"

	def add_to_foe(self,foe_builder):
		foe_builder.target.behav = self.text


class XBuilder(TextBuilder):
	element_name = "x"

	def add_to_foe(self,foe_builder):
		foe_builder.target.x = get_eval(self.text)


class YBuilder(TextBuilder):
	element_name = "y"

	def add_to_foe(self,foe_builder):
		foe_builder.target.y = get_eval(self.text)


class BulletBuilder(TextBuilder):
	element_name="bullet"

	def add_to_foe(self,foe_builder):
		foe_builder.target.bullet = self.text
			

class SpriteBuilder(TextBuilder):
	element_name="sprite"

	def add_to_foe(self,foe_builder):
		foe_builder.target.sprite = self.text


class FrameBuilder(TextBuilder):
	element_name="frame"

	def add_to_foe(self,foe_builder):
		foe_builder.target.frame = get_eval(self.text)


#########################
## Parsing and Building

current_object_stack = []

builder_classes = {
	'stage'           : StageBuilder,
	'foe'             : FoeBuilder,
	'behav'           : BehavBuilder,
	'x'               : XBuilder,
	'y'               : YBuilder,
	'frame'           : FrameBuilder,
	'bullet'          : BulletBuilder,
	'sprite'          : SpriteBuilder,
	}
					 

class StageHandler( xml.sax.handler.ContentHandler ):
	def characters( self, chars ):
		if current_object_stack:
			current_object = current_object_stack[-1]
			current_object.add_text( chars.strip() )

	def startDocument( self ):
		namespace = None

	def endDocument( self ):
		if namespace is None: 
			l.warning( "No foe found" )

	def startElement( self, name, attrs ):
		if name in builder_classes:
			builder = builder_classes[name]()
			current_object_stack.append( builder )
			#uilder.add_attrs(attrs) # does nothng if element doesn't like attrs
		else:
			l.warning( "Unknown element : " + name )

	def startElementNS( self, name, qname, attrs ):
		print name, qname, attrs

	def endElement( self, name ):
		if name in builder_classes:
			try:
				builder = current_object_stack.pop()
				if current_object_stack:
					parent_builder = current_object_stack[-1]
					builder.add_to( parent_builder )
			except:
				raise

#FIXME: find a slightly less moronic name
myStageHandler = StageHandler()

myParser = xml.sax.make_parser()
myParser.setFeature(xml.sax.handler.feature_validation, False)
myParser.setFeature(xml.sax.handler.feature_external_ges, False)
myParser.setContentHandler(myStageHandler)


def get_action( name ):
	global namespace
	try:
		f = open( name, 'r' )
		myParser.parse(f)
		f.close()
	except Exception,ex:
		l.error( "Error while parsing Stage file : " + str(name) )
		l.debug("Exception :" + str(ex))
		raise
	return copy.deepcopy(namespace)


class StageController:
	
	def set_behavior( self, name ):
		self.action = get_action(name)

	def run(self):
		self.action.run(self)
		
	def set_game_object(self, game_object):
		self.game_object = game_object
