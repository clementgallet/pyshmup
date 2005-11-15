import random
import logging
import re
import xml.sax, xml.sax.handler
import copy

console = logging.StreamHandler( )
console.setLevel( logging.DEBUG )
formatter = logging.Formatter( '%(name)-12s: %(levelname)-8s %(message)s' )
console.setFormatter( formatter )

logging.getLogger( "bulletml" ).addHandler( console )
l = logging.getLogger( "bulletml" )

#principe :

#  parcours SAX du fichier xml
#  creation des objets par appel aux constructeurs respectifs (voire constructeur minimal puis compsition) puis enregistrement en factory pour tous les objets référençables. en effet, une formule (ou Value) peut très bien instancée directement, puisque qu'elle sera stockée dans un état neutre dans les éléments de la factory
#  on "instancie" ensuite les objets effectivement utilisés par un appel à la Factory correspondante (avec son label)


f = open( "bee.xml", "r" )

tobuild_stack = []
children_stack = []

known_elements = [ "action", "changeSpeed", "speed" , "term", 
          "changeDirection", "direction", "repeat", "fire", "wait", "bullet" ]


def get_bullet(filename):
	pass

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
			if game_object_control.turn_status == END_TURN:
				break
			
# should implement a generic interface to subactions
#   something like .update()

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

class Direction:
	type=0
	value=0

class Speed:
	type=0
	value=0

class Horizontal:
	type=0
	value=0

class Vertical:
	type=0
	value=0

class Term:
	value=0

class Times:
	value=0

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

class Null:
	def run( self ):
		pass

class Param:
	"""Params are replaced by Values placed in actionRef, bulletRef, and fireRef.
	"""

HEUR_VALID_FORMULA = re.compile(r'^([0-9]|\$(rand|rank|[0-9])|\+|-)*\$')

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
			logging.error( 'Invalid formula : ' + formula )
			formula='0'
		self.params = params
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
		











#def buildTerm( ):
#  obj = object_stack.pop( )
#  while obj != "term":
#     print obj
# adopt visitor pattern ?

# $visitor calls $child.add_to_$($visitor.TYPE), passing himself as the argument on which to operate

def get_base( name ):
	return { name : name }



class Builder:
	def __init__( self, typename ):
		self.target = typename + 'Builder'

	def add_to( self, builder ):
		l.debug( self.target + '.add_to( ' + builder.target + ' )' )

	def add_text( self, text ):
		if text:
			l.warning( "ignoring text : " + text )


def get_builder( name ):
	builders = { 'bullet' : BulletRef(), # might work like this : when adding
	  #non params to a BulletRef builder (ie. Bullut childs actually), add it to the Refs'
	  # target ; create random name at build time if needs be
	             'action' : ActionRef(),
					 }
	# no, pleaso separate Builder from finished object (cf. endElement)
	return Builder( name )
	             


class SpeedBuilder( Builder ):
	formula = ''
	
	def add_text(self, text):
		self.formula += text

	def add_to_ChangeSpeed(self, changespeed_builder):
		changespeed_builder.speed = Value( self.formula )

class DirectionBuilder( Builder ):
	formula = ''
	
	def add_text(self, text):
		self.formula += text

	def add_to_ChangeSpeed(self, changedirection_builder):
		changedirection_builder.direction = Value( self.formula )

class ChangeSpeedBuilder( Builder ):
	target = ChangeSpeed()
	
	def build(self):
		self.target.term = self.term
		self.target.speed = self.term
		return self.target

class TermBuilder( Builder ):
	formula = ''
	
	def add_text(text):
		self.formula += text

	def add_to_ChangeDirection(self, changedirection_builder):
		changedirection_builder.term = Value( self.formula )

	def add_to_ChangeSpeed(self, changespeed_builder):
		changespeed_builder.term = Value( self.formula )

class WaitBuilder( Builder ):
	target = Wait( )

	def build(self):
		#FIXME: error handling
		self.target.term = self.term
		return self.target

	



# warning: this is so wrong
#  finding a safe path to obtain this state is left as an exercice
#  to the dumber readers

#        |            |    |             |
#        |            |    | <direction> |
#        |            |    |             |
#        |            |    |   <speed>   |
#        |            |    |             |
#        |            |    |   "fire"    |
#        |            |    |             |
#        |   "fire"   |    |   <action>  |
#        |            |    |             |
#        | "bulletml" |    |  "bulletml" |
#        \____________/    \_____________/
#
#         tobuild_stack     children stack

# in chilren stack : objects appearing before the first mark are the 
# current object's chidren. construction of current object must return
# after removing that mark






current_object_stack = []

class BulletMLParser( xml.sax.handler.ContentHandler ):
	def characters( self, chars ):
		if current_object_stack:
			current_object = current_object_stack[-1]
			current_object.add_text( chars.strip() )

	def startDocument( self ):
		print "Document begins"

	def endDocument( self ):
		print "Document ends"

	def startElement( self, name, attrs ):
#		print "start " + name
		if name in known_elements:
			builder = get_builder( name )
			current_object_stack.append( builder )
		
		else:
			l.warning( "Unknown element : " + name )

	def startElementNS( self, name, qname, attrs ):
		print name, qname, attrs

	def endElement( self, name ):
		# call an object builder (composition) that will pop the stack until finding its mark
		# adding popped stuff to the object in the meantime
#		print "end " + name
		if name in known_elements:
			try:
				builder = current_object_stack.pop()
				if current_object_stack:
					parent_builder = current_object_stack[-1]
					builder.add_to( parent_builder )
			except:
				l.error( "Don't know what to do with " + name + " element." )
#				raise

#FIXME: find a slightly less moronic name
myBulletMLParser = BulletMLParser()

def get_main_action( name ):
	#FIXME: error handling anyone ?
	if not name in behaviors:
		set_action_namespace( name )
		f = open( name, 'r' )
		xml.sax.parse( f, myBulletMLParser )
		f.close()
		# actions have been registered
		behaviors[name] = 
		
class GameObjectController:
	game_object = None
	
	def set_behavior( self, name ): # name is really a namepace
		self.master_action = get_main_action( name )
