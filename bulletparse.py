# -*- coding: utf-8 -*-

import random
import logging
import re
import xml.sax, xml.sax.handler
import copy

############
## Logging

console = logging.StreamHandler( )
console.setLevel( logging.DEBUG )
formatter = logging.Formatter( '%(name)-12s: %(levelname)-8s %(message)s' )
console.setFormatter( formatter )

logging.getLogger('').addHandler( console )
l = logging.getLogger( "bulletml" )
l.setLevel( logging.DEBUG )

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

# run() status codes
BREAK = 0 # returned by Wait
CONTINUE = 1 # action is not yet finished
DONE = 2
# those are cummunicated via game_object_control.turn_status

class Control:
	def run_first(self, game_object_control, params=[]):
		pass

	def run_not_first(self, game_object_control, params=[]):
		l.error( str(type(self)) + " does not implement .run()" ) # actually run_not_first, but makes more sense to outside reader ?
	
	def run(self, game_object_control, params=[]):
		self.run_first(self, game_object_control, params)
		self.run = self.run_not_first
		self.run(game_object_control, params)

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
		is_done = True # -> False if a subaction is not finished
		for child in self.subactions:
			child.run( game_object_control, params )
			if game_object_control.turn_status == CONTINUE:
				is_done = False
			elif game_object_control.turn_status == WAIT:
				is_done = False
				break
		if is_done:
			game_object_control.turn_status = DONE
		game_object_control.turn_status = CONTINUE
			
class Fire:
	label=''
	direction=None
	speed=None
	bullets=0 # bulletRefs

class ChangeDirection(Control):
	# has two values :
	#  term and direction

	def run_not_first(self, game_object_control, params = []):
		if self.frame >= self.term:
			return DONE
		self.frame += 1
		game_object_control.game_object.set_direction( self.initial_direction + \
				(self.frame / self.term) * self.direction_offset )
		return CONTINUE

	def run_first(self, game_object_control, params=[]):
		self.initial_direction = game_object_control.get_direction()
		self.direction_offset = self.direction_value.get( params ) - self.initial_direction
		if (self.direction_offset % 360) > 180:
			self.direction_offset = - self.direction_offset
		self.frame = 0
		self.term = self.term_value.get( params )

class ChangeSpeed:
	# has two values :
	#  term and speed

	def run_not_first(self, game_object_control, params = []):
		if self.frame >= self.term:
			game_object_control.turn_status = DONE
		else:
			self.frame += 1
			game_object_control.game_object.set_speed( self.initial_speed + \
		  		(self.frame / self.term) * self.speed_offset )
			game_object_control.turn_status = CONTINUE

	def run_first(self, game_object_control, params = []):
		self.initial_speed = game_object_control.get_direction()
		self.speed_offset = self.speed_value.get( params )
		self.frame = 0
		self.term = self.term_value.get( params )


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

	def run_first(self, game_object_control, params=[]):
		self.term = self.term_value.get(params)
		self.frame = 0

	def run_not_first(self, game_object_control, params=[]):
		if self.frame > self.term:
			game_object_control.turn_status = DONE
		else:
			self.frame += 1
			game_object_control.turn_status = WAIT

class Vanish:
	pass

# should repeat : someting,term=n ; wait,term=n ; somethingelse,term=m
# exec somethingelse at all ?
#  actually this is a problem for Action.run_not_first

class Repeat:
	times=0
	actionref=0

	def run_not_first(self, game_object_control, params):
		if self.repetition > self.times:
			game_object_control.turn_status = DONE
		res = self.action_ref.run(game_object_control, params)
		if res == DONE:
			self.repetition += 1
			if self.repetition != self.times:
				self.actionref.init()

	def run_first(self, game_object, params):
		self.times = self.times_value.get( params )
		self.repetition = 0
	

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
	actionname=''
	param_values=[]

	def init(self, params=[]):
		self.actionref = get_action(namespace, actionname)
		self.params = [val.get(params) for val in self.params_values]

	def run_first(self, game_object_control, params=[]):
		self.init(params)

	def run_not_first(self, game_object_control, params = []):
		self.action.run(game_object_control, self.params)


class FireRef:
	namespace = ''
	label=''
	fire_params=[]







###########
## Values

HEUR_VALID_FORMULA = re.compile(r'^([0-9.]|\$(rand|rank|[0-9])|\+|-|/|\*)*$')

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

# find appropriate name
main_action = None
current_namespace = "pie !"

def register_action(ns, name, action):
	namespaces[ns]['action'][name] = action

def register_fire(ns, name, fire):
	namespaces[ns]['fire'][name] = fire

def register_bullet(ns, name, bullet):
	namespaces[ns]['bullet'][name] = bullet

# should I encapsulate all of this in a single object ? is there a point ?

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
			l.debug( "Ignoring text : " + text + " in " + self.element_name + "." )


class FormulaBuilder(Builder):
	formula = ''
	def add_text(self, text):
		self.formula += text

class SubActionBuilder:
	def add_to_action(self, action_builder):
		action_builder.target.subactions.append(self.target)

class RefBuilder:
	def __init__(self): # FIXME: overloading __init__ while planning 
	                    # on using multiple inheritance is just asking
							  # for trouble ; oh well..
		self.target.namespace = current_namespace




class BulletmlBuilder(Builder):
	element_name =  "bulletml"


class BulletBuilder(Builder):
	element_name="bullet"
	target = Bullet()

	def add_to_bulletml(self, bulletml_builder):
		global main_action
		register_bullet(current_namespace, self.label, self.target)
		if not main_action:
			main_action = self.target


class ActionBuilder(Builder):
	element_name="action"
	target = Action()

	def add_to_bulletml(self, bulletml_builder):
		global main_action
		register_action(current_namespace, self.label, self.target)
		if not main_action:
			main_action = self.target


class FireBuilder(Builder):
	element_name="fire"
	target = Fire()

	def add_to_bulletml(self, bulletml_builder):
		global main_action
		register_fire(current_namespace, self.label, self.target)
		if not main_action:
			main_action = self.target


class ChangeDirectionBuilder(Builder, SubActionBuilder):
	element_name="changeDirection"
	target = ChangeDirection


class ChangeSpeedBuilder(Builder, SubActionBuilder):
	element_name="changeSpeed"
	target = ChangeSpeed()
	

class AccelBuilder(Builder):
	element_name="accel"
	target = Accel( )

	def build(self):
		self.target.term = self.term
		self.target.horizontal = self.horizontal
		self.target.vertical = self.vertical


class WaitBuilder(Builder, SubActionBuilder):
	element_name="wait"
	target = Wait()


class VanishBuilder(Builder):
	element_name="vanish"


class RepeatBuilder(Builder):
	element_name="repeat"
	target = Repeat()


class DirectionBuilder(FormulaBuilder):
	element_name="direction"
	
	def add_to_changeDirection(self, changedirection_builder):
		changedirection_builder.direction_value = Value( self.formula )


class SpeedBuilder(FormulaBuilder):
	element_name="speed"

	def add_to_changeSpeed(self, changespeed_builder):
		changespeed_builder.target.speed_value = Value( self.formula )


class HorizontalBuilder(FormulaBuilder):
	element_name="horizontal"

	def add_to_accel(self, accel_builder):
		accel_builder.target.horizontal_value = Value( self.formula )


class VerticalBuilder(FormulaBuilder):
	element_name="vertical"

	def add_to_accel(self, accel_builder):
		accel_builder.target.vertical_value = Value( self.formula )


class TermBuilder(FormulaBuilder):
	element_name="term"

	def add_to_changeDirection(self, changedirection_builder):
		changedirection_builder.target.term_value = Value( self.formula )

	def add_to_changeSpeed(self, changespeed_builder):
		changespeed_builder.target.term_value = Value( self.formula )

	def add_to_accel(self, accel_builder):
		accel_builder.target.term_value = Value( self.formula )


class TimesBuilder(FormulaBuilder):
	element_name="times"

	def add_to_repeat(self, repeat_builder):
		repeat_builder.target.times_value = Value(self.formula)


class BulletRefBuilder(Builder):
	element_name="bulletRef"


class ActionRefBuilder(Builder, RefBuilder):
	element_name="actionRef"
	target = ActionRef()

	def add_to_repeat(self, repeat_builder):
		repeat_builder.target.target = self.target # clumsy


class FireRefBuilder(Builder):
	element_name="fireRef"


class ParamBuilder(Builder):
	element_name="param"

	def add_to_actionRef(self, actionref_builder):
		actionref_builder.target.param_values.append(self.value)

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

		namespaces[current_namespace] = { 'action' : {},
		                                  'fire'   : {},
													 'bullet' : {} }

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
			try:
				builder.label = attrs.getValue('label')
			except:
				pass
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
			l.debug(ex)
			raise
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
