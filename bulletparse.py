# -*- coding: utf-8 -*-

import random
import logging
import re
import xml.sax, xml.sax.handler
import copy

console = logging.StreamHandler( )
console.setLevel( logging.DEBUG )
formatter = logging.Formatter( '%(name)-12s: %(levelname)-8s %(message)s' )
console.setFormatter( formatter )

logging.getLogger('').addHandler( console )
l = logging.getLogger( "bulletml" )

#principe :

#  parcours SAX du fichier xml
#  creation des objets par appel aux constructeurs respectifs (voire constructeur minimal puis compsition) puis enregistrement en factory pour tous les objets référençables. en effet, une formule (ou Value) peut très bien instancée directement, puisque qu'elle sera stockée dans un état neutre dans les éléments de la factory
#  on "instancie" ensuite les objets effectivement utilisés par un appel à la Factory correspondante (avec son label)





# FIXME: *gasp*
class NullAction:
	def run( self ):
		pass

# namespaces[namespace]['action'|'fire'|'bullet'][label] = <controller_object>
namespaces = { "null" : { 'action' : {}, 'fire' : {}, 'bullet' : {} } }

main_actions = { "null" : NullAction() }

#######################
## Controller objects

class BulletML:
	type=''
	contents=[]

class Bullet:
	label=''
	direction=None
	speed=None
	actions=[] # actionRefs

# if an action is found in a file, build the action separately and ref it

class Action:
	label=''
	subactions=[]

	def run(self, game_object_control, params = []):
		for child in self.subactions:
			child.run( game_object_control, params )
			if game_object_control.turn_status == WAIT:
				break
			game_object_control.turn_status = GO_ON
			
class Fire:
	label=''
	direction=None
	speed=None
	bullets=0 # bulletRefs

class ChangeDirection:
	# has two values :
	#  term and direction

	def run(self, game_object_control, params = []):
		try:
			self.frame += 1
			if self.frame > self.term:
				return
			else:
				game_object_control.game_object.set_direction( self.initial_direction + \
				      (self.frame / self.term) * self.direction_offset )
				return
		except: # first run
			self.initial_direction = game_object_control.get_direction()
			self.direction_offset = self.direction.get( params ) - self.initial_direction
			if (self.direction_offset % 360) > 180:
				self.direction_offset = - self.direction_offset
			self.frame = 0
			# FIXME: this code is rather misleading
			self.term = self.term.get( params )
			self.run( game_object_control )

class ChangeSpeed:
	# has two values :
	#  term and speed

	def run(self, game_object_control, params = []):
		try:
			self.frame += 1
			if self.frame > self.term:
				return
			else:
				game_object_control.game_object.set_speed( self.initial_speed + \
				      (self.frame / self.term) * self.speed_offset )
				return
		except: # first run
			self.initial_speed = game_object_control.get_direction()
			self.speed_offset = self.speed.get( params )
			self.frame = 0
			self.term = self.term.get( params )
			self.run( game_object_control )


# random note : when an action is represented as an actionRef, special
# precautions should be taken so that the actionRef's params are transfered
# in contrast, a «real» actionRef shoud not tranfer its given params
# ==> have a flag
	
	

class Accel:
	horizontal=0
	vertical=0
	term=0

# note: correct way to build those is obviously to read full xml element then
# construct object
#  also, interpret formulas while collecting data
#   but then what about params ?
#    it's ok, params are children, so are fully built and calculated while
#    reading. should build a list of them
#     uh-oh.. WRONG. $rand and co. need te be recomputed each time. on the
#     other hand, params are fixed at call time so... obviously we can't
#     preinstantiate fully objects at read-time. I suggest a 
#     Value/ComputedValue separation : Value.get_value() yields a 
#     ComputedValue, eventually from formula. if value is not subject to
#     change, ie. not a param in .*Ref or in a sequence...
#      nope, what about sequence[changeSpeed($rand),wait(3)] ?
#      formulas will be left intact for now
#       only, when passing parameters, use Value.get

class Wait:
	duration=0

class Vanish:
	pass

class Repeat:
	times=0
	action=0

# when .run()ed, objects are passed a reference to the foe, bullet, etc..

# Refs should store an instantiated object after the first call
#  actually I should stock ready-to run versions of all named objects
#  and clone them to instantiate them
#  ==> factory
#   what's more, Refs need to keep a namespace identifier. it might be
#   set to current namespace upon bulding

class BulletRef:
	namespace = ''
	label=''
	bullet_params=[]

class ActionRef:
	namespace = ''
	label=''
	action_params=[]

class FireRef:
	namespace = ''
	label=''
	fire_params=[]







###########
## Values

HEUR_VALID_FORMULA = re.compile(r'^([0-9.]|\$(rand|rank|[0-9])|\+|-)*$')

filter_definitions = [
  ( re.compile(r'\$rand'), '(random.random())' ),
  ( re.compile(r'\$rank'), '(0.5)'),
  ( re.compile(r'\$([0-9]+)'), r'(self.params[\1-1])') ]

def substitute(r,repl):
	# help fight the lambda abuses ! join now !
	def aux(formula):
		return r.sub(repl, formula)
	return aux

formula_filters = [ substitute(r, repl)
                   for r, repl in filter_definitions ]

class Value:
	"""Value uses a BulletML formula and an optional list of parameters.
	
		A value is recomputed for each call of .get(), including $rand.
		Things like :
		  <changeSpeed>
		    <speed> 2+2*$rand </speed>
			 <term> 10 </term>
		  </changeSpeed>
		should therefore be avoided ? Use a parameterized action instead.
	"""
	def __init__(self, formula):
		formula.replace( '\n', '' )
		if not HEUR_VALID_FORMULA.match(formula):
			l.error( 'Invalid formula : ' + formula )
			formula='0'
		old_formula = ''
		while formula != old_formula:
			old_formula = formula
			for f in formula_filters:
				formula = f(formula)
		try:
			eval(formula)
		except:
			logging.error( 'Invalid formula, interpreted as : ' + formula )
			formula='0'
		self.formula=formula

	def get(self, params=[]):
		return eval(self.formula)
		

def get_fire( namespace, label ):
	return namespaces[namespace]['fire'][label]







#############
## Builders

main_action = None
current_namespace = "pie !"


class Builder(object):
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
			l.debug( "Ignoring text : " + text )


class BulletmlBuilder(Builder):
	element_name =  "bulletml"


class BulletBuilder(Builder):
	element_name="bullet"


class ActionBuilder(Builder):
	element_name="action"


class FireBuilder(Builder):
	element_name="fire"


class ChangeDirectionBuilder(Builder):
	element_name="changeDirection"


class ChangeSpeedBuilder( Builder ):
	element_name="changeSpeed"
	target = ChangeSpeed()
	
	def build(self):
		self.target.term = self.term
		self.target.speed = self.term
		return self.target


class AccelBuilder(Builder):
	element_name="accel"


class WaitBuilder( Builder ):
	element_name="wait"
	target = Wait( )

	def build(self):
		#FIXME: error handling
		self.target.term = self.term
		return self.target


class VanishBuilder(Builder):
	element_name="vanish"


class RepeatBuilder(Builder):
	element_name="repeat"


class DirectionBuilder( Builder ):
	element_name="direction"
	formula = ''
	
	def add_text(self, text):
		self.formula += text

	def add_to_changeSpeed(self, changedirection_builder):
		changedirection_builder.direction = Value( self.formula )


class SpeedBuilder( Builder ):
	element_name="speed"
	formula = ''
	
	def add_text(self, text):
		self.formula += text

	def add_to_changeSpeed(self, changespeed_builder):
		changespeed_builder.speed = Value( self.formula )


class HorizontalBuilder(Builder):
	element_name="horizontal"


class VerticalBuilder(Builder):
	element_name="vertical"


class TermBuilder( Builder ):
	element_name="term"
	formula = ''
	
	def add_text(self, text):
		self.formula += text

	def add_to_changeDirection(self, changedirection_builder):
		changedirection_builder.term = Value( self.formula )

	def add_to_changeSpeed(self, changespeed_builder):
		changespeed_builder.term = Value( self.formula )


class TimesBuilder(Builder):
	element_name="times"


class BulletRefBuilder(Builder):
	element_name="bulletRef"


class ActionRefBuilder(Builder):
	element_name="actionRef"


class FireRefBuilder(Builder):
	element_name="fireRef"


class ParamBuilder(Builder):
	element_name="param"

builders = { 
					 'bulletml' : BulletmlBuilder(),
					 'bullet' : BulletBuilder(),
					 'action' : ActionBuilder(),
					 'fire' : FireBuilder(),
					 'changeDirection' : ChangeDirectionBuilder(),
					 'changeSpeed' : ChangeSpeedBuilder(),
					 'accel' : AccelBuilder(),
					 'wait' : WaitBuilder(),
					 'vanish' : VanishBuilder(),
					 'repeat' : RepeatBuilder(),
					 'direction' : DirectionBuilder(),
					 'speed' : SpeedBuilder(),
					 'horizontal' : HorizontalBuilder(),
					 'vertical' : VerticalBuilder(),
					 'term' : TermBuilder(),
					 'times' : TimesBuilder(),
					 'bulletRef' : BulletRefBuilder(),
					 'actionRef' : ActionRefBuilder(),
					 'fireRef' : FireRefBuilder(),
					 'param' : ParamBuilder(),
					 }




#########################
## Parsing and Building

current_object_stack = []

class BulletMLParser( xml.sax.handler.ContentHandler ):
	def characters( self, chars ):
		if current_object_stack:
			current_object = current_object_stack[-1]
			current_object.add_text( chars.strip() )

	def startDocument( self ):
		global main_action
		main_action = None

		namespaces[current_namespace] = {}

	def endDocument( self ):
		global main_action
		if not main_action:
			main_action = NullAction()
			l.warning( "No main action found in " + current_namespace )
		main_actions[current_namespace] = main_action
		print main_actions
		main_action = None

	def startElement( self, name, attrs ):
		if name in builders:
			builder = copy.copy( builders[name] )
			current_object_stack.append( builder )
		
		else:
			l.warning( "Unknown element : " + name )

	def startElementNS( self, name, qname, attrs ):
		print name, qname, attrs

	def endElement( self, name ):
		if name in builders:
			try:
				builder = current_object_stack.pop()
				if current_object_stack:
					parent_builder = current_object_stack[-1]
					builder.add_to( parent_builder )
			except:
				raise

#FIXME: find a slightly less moronic name
myBulletMLParser = BulletMLParser()

def set_action_namespace( name ):
	global current_namespace
	current_namespace = name

def get_main_action( name ):
	if not name in main_actions:
		set_action_namespace( name )
		try:
			f = open( name, 'r' )
			xml.sax.parse( f, myBulletMLParser )
			f.close()
		except Exception,ex:
			l.error( "Error while parsing BulletML file : " + name )
			return main_actions["null"]
	return main_actions[name]
		
class GameObjectController:
	game_object = None
	
	def set_behavior( self, name ): # name is really a namepace
		self.master_action = get_main_action( name )


############
## Testing

if __name__ == '__main__':
	ctrl = GameObjectController()

	ctrl.set_behavior( 'bee.xml' )
