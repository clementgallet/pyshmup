# -*- coding: utf-8 -*-

import random
import logging
import re
import xml.sax, xml.sax.handler
import copy

import traceback

RANK = 1.0

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

#principe :

#  parcours SAX du fichier xml
#  creation des objets par appel aux constructeurs respectifs (voire constructeur minimal puis compsition) puis enregistrement en factory pour tous les objets référençables. en effet, une formule (ou Value) peut très bien instancée directement, puisque qu'elle sera stockée dans un état neutre dans les éléments de la factory
#  on "instancie" ensuite les objets effectivement utilisés par un appel à la Factory correspondante (avec son label)





# FIXME: *gasp*
class NullAction:
	def run(self, game_object_control, params=[]):
		pass

# namespaces[namespace]['action'|'fire'|'bullet'][label] = <controller_object>
namespaces = { "null" : { 'action' : {}, 'fire' : {}, 'bullet' : {}, 'main_actions' : [] } }

def get_action(namespace, label):
	return copy.deepcopy( namespaces[namespace]['action'][label] )

def get_fire(namespace, label):
	return copy.deepcopy( namespaces[namespace]['fire'][label] )

def get_bullet(namespace, label):
	return copy.deepcopy( namespaces[namespace]['bullet'][label] )







#######################
## Controller objects

# run() status codes
WAIT = 0 # returned by Wait
CONTINUE = 1 # action is not yet finished
DONE = 2
# those are communicated via game_object_control.turn_status

class Control:
	# A word about copy.deepcopy : it does not support copying functions,
	# and so chokes on objects that have dynamically bound methods.
	# DO NOT TRY TO DEEPCOPY A CONTROL THAT HAS BEEN .RUN()ED !
	# It is actually cool since it sort of warns us about tampering with 
	# namespaces[]'s content.
	def run_first(self, game_object_control, params=[]):
		pass

	def run_not_first(self, game_object_control, params=[]):
		l.debug( str(self) + " does not implement .run_not_first()" ) 

	def run(self, game_object_control, params=[]):
		self.run_first(game_object_control, params)
		self.run = self.run_not_first
		self.run(game_object_control, params)
		
	# We can't really always restart with a "pure" control (ie. from namespaces)
	# because of the sequence-type speeds and directions. Therefore we need
	# explicit restarting, sometimes preserving part of the state.
	# This has a few consequences :
	#  - run_first should allow multiple calls, or reinit() must be overloaded.
	#  - the original state should never really be forgotten (namely variables
	#    set by Builders.
	def reinit(self, game_object_control, params=[]):
		self.run_first(game_object_control, params)
		

class Bullet(Control):
	def __init__(self):
		self.subactions = []

	def run_first(self, game_object_control, params=[]):
		try:
			game_object_control.game_object.direction = \
			   self.direction.get_standard(game_object_control, params)
		except AttributeError:
			pass
		try:
			game_object_control.game_object.speed = self.speed.get_standard(params)
		except AttributeError:
			pass

	def run_not_first(self, game_object_control, params=[]):
		is_done = False
		for child in self.subactions:
			child.run( game_object_control, params )
			if game_object_control.turn_status == CONTINUE:
				is_done = False
			elif game_object_control.turn_status == WAIT:
				is_done = False
				break
		if is_done:
			game_object_control.turn_status = DONE
		else:
			game_object_control.turn_status = CONTINUE

# if an action is found in a file, build the action separately and ref it

class Action(Control):
	def __init__(self):
		self.subactions = []

	def run_not_first(self, game_object_control, params = []):
#		is_done = True # -> False if a subaction is not finished
#		for child in self.subactions:
#			child.run( game_object_control, params )
#			if game_object_control.turn_status == CONTINUE:
#				is_done = False
#			elif game_object_control.turn_status == WAIT:
#				is_done = False
#				break
#		if is_done:
#			game_object_control.turn_status = DONE
#		else:
#			game_object_control.turn_status = CONTINUE
		if self.current_index >= self.sub_length:
			game_object_control.turn_status = DONE
		else:
			self.subactions[self.current_index].run(game_object_control, params)
			if game_object_control.turn_status == DONE:
				self.current_index += 1
			if self.current_index >= self.sub_length:
				game_object_control.turn_status = DONE
			else:
				game_object_control.turn_status = CONTINUE
			

	def run_first(self, game_object_control, params=[]):
		self.current_index = 0
		self.sub_length = len(self.subactions)

	def reinit(self, game_object_control, params=[]):
		for child in self.subactions:
			child.reinit(game_object_control, params)
		self.current_index = 0

class Fire(Control):
	def reinit_launch(self, game_object_control, params=[]):
		"""Builds the argument list for fire()."""
		control = GameObjectMainController()
		sub_control = GameObjectSubController()
#		self.bulletref.reinit(game_object_control, params=[])
		sub_control.top_action = copy.deepcopy(self.bulletref)
		control.sub_controllers = [sub_control]
		self.kwargs = { 'controller' : control }
	

	def run_first(self, game_object_control, params=[]):
		self.already_fired = False
		self.reinit_launch(game_object_control, params)

	def run_not_first(self, game_object_control, params=[]):
		if not self.already_fired:
			self.already_fired = True

			main_control = GameObjectMainController()
			main_control.game_object = game_object_control.game_object

			bullet = self.bulletref.get_bullet()

			for action in bullet.subactions:
				main_control.add_action(action)

			# arbitrary order of precedence :
			#  bullet < fire
			# (if that turns out to be false, swap the lines ;) )
			# we try get a Speed and Direction from them

			# but wait, there's more !
			# Fire receives params, and if bulletRef.is_real_ref, bulletRef
			# has its own too, whose actual values might depend en Fire's
			# received params. confused yet ?

			# most of this code may be sent to run_first ?

			NOWHERE = 0
			BULLET = 1
			FIRE = 2

			speed_location = NOWHERE
			direction_location = NOWHERE

			try:
				speed = bullet.speed
				speed_location = BULLET
			except: pass
			try:
				direction = bullet.direction
				direction_location = BULLET
			except: pass

			try:
				speed = self.speed
				speed_location = FIRE
			except: pass
			try:
				direction = self.direction
				direction_location = FIRE
			except: pass

			self.bulletref.reinit_params(params)

			if speed_location == FIRE:
				numeric_speed = speed.get_standard(game_object_control, params)
			elif speed_location == BULLET:
				numeric_speed = speed.get_standard(game_object_control, self.bulletref.params)
			else:
				numeric_speed = game_object_control.game_object.speed
				
			if direction_location == FIRE:
				numeric_direction = direction.get_standard(game_object_control, params)
			elif direction_location == BULLET:
				numeric_direction = direction.get_standard(game_object_control, self.bulletref.params)
			else:
				numeric_direction = game_object_control.game_object.direction

			if self.bulletref.is_real_ref:
				main_control.params = self.bulletref.params
			else:
				main_control.params = params

			main_control.game_object.fire(main_control, numeric_direction, numeric_speed)

			game_object_control.last_speed = numeric_speed
			game_object_control.last_direction = numeric_direction

		game_object_control.turn_status = DONE
			
	# default reinit


class ChangeDirection(Control):
	# has two values :
	#  term and direction

	def run_not_first(self, game_object_control, params = []):
		if self.frame >= self.term:
			game_object_control.turn_status =  DONE
		else:
			self.frame += 1
			game_object_control.game_object.direction = self.initial_direction + \
			   (float(self.frame) / self.term) * self.direction_offset
			game_object_control.turn_status =  CONTINUE

	def run_first(self, game_object_control, params=[]):
		self.initial_direction = game_object_control.game_object.direction
		self.direction_offset = self.direction.get_standard(game_object_control, params) - \
		                          self.initial_direction
		self.direction_offset = self.direction_offset % 360
		if abs(self.direction_offset) > 180:
			self.direction_offset -= 360
		self.frame = 0
		self.term = self.term_value.get( params )

class ChangeSpeed(Control):
	# has two values :
	#  term and speed

	def run_not_first(self, game_object_control, params = []):
		if self.frame >= self.term:
			game_object_control.turn_status = DONE
		else:
			self.frame += 1
			game_object_control.game_object.speed = \
			       self.get_new_speed(game_object_control, params)
			game_object_control.turn_status = CONTINUE

	def run_first(self, game_object_control, params = []):
		initial_speed = game_object_control.game_object.speed
		self.frame = 0
		self.term = int(self.term_value.get( params ))
		if self.term <= 0:
			self.term = 1
			l.error("Invalid term in speed : " + self.speed.formula)

		if self.speed.type == "absolute":
			self.theoretical_speed = initial_speed
			self.speed_offset = float( self.speed.get(params) -
			                       initial_speed ) / self.term
			self.get_new_speed = self.get_new_speed_absolute
		elif self.speed.type == "relative":
			self.theoretical_speed = initial_speed
			self.speed_offset = float( self.speed.get(params) ) / self.term
			self.get_new_speed = self.get_new_speed_relative
		else: # "sequence"
			self.last_speed = game_object_control.game_object.speed
			self.get_new_speed = self_get_new_speed_sequence

	def get_new_speed_absolute(self, game_object_control, params=[]):
		self.theoretical_speed += self.speed_offset
		return self.theoretical_speed

	def get_new_speed_relative(self, game_object_control, params=[]):
		self.theoretical_speed += self.speed_offset
		return self.theoretical_speed
		

	def get_new_speed_sequence(self, game_object_control, params=[]):
		# we assume it will be used to fire() (thus fed back)
		return game_object_control.last_speed + self.get(params)

# FIXME: implement following remark
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

class Wait(Control):
	duration=0

	def run_first(self, game_object_control, params=[]):
		self.term = self.term_value.get(params)
		self.frame = -1

	def run_not_first(self, game_object_control, params=[]):
		if self.frame > self.term:
			game_object_control.turn_status = DONE
		else:
			self.frame += 1
			game_object_control.turn_status = WAIT

class Vanish(Control):
	def run_first(self, game_object_control, params=[]):
		game_object_control.game_object.vanish()

	def run_not_first(self, game_object_control, params=[]):
		game_object_control.turn_status = DONE

# should repeat : someting,term=n ; wait,term=n ; somethingelse,term=m
# exec somethingelse at all ?
#  actually this is a problem for Action.run_not_first

class Repeat(Control):
	def run_not_first(self, game_object_control, params):
		if self.repetition > self.times:
			game_object_control.turn_status = DONE
		else:
			self.actionref.run(game_object_control, params)
			if game_object_control.turn_status == DONE:
				self.repetition += 1
				if self.repetition != self.times:
					self.actionref.reinit(game_object_control, params)
			game_object_control.turn_status = CONTINUE

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

class BulletRef(Control):
	def __init__(self):
		self.param_values = []
		self.is_real_ref = True

	def reinit_params(self, params):
		self.params = [val.get(params) for val in self.param_values]

	def get_bullet(self):
		return get_bullet(self.namespace, self.label)

	def run_first(self, game_object_control, params=[]):
		self.bullet = get_bullet(self.namespace, self.label)
		# Do not .reinit_params() ! Only Fire knew the appropriate params.

	def run_not_first(self, game_object_control, params=[]):
		# If real ref, has its own params to transmit.
		if self.is_real_ref:
			self.bullet.run(game_object_control, self.params)
		else:
			self.bullet.run(game_object_control, params)
	

class ActionRef(Control):
	def __init__(self):
		self.param_values = []
		self.is_real_ref = True

	def reinit(self, game_object_control, params=[]):
		self.reinit_params(params)
		self.action.reinit(game_object_control, self.params)

	def reinit_params(self, params):
		self.params = [val.get(params) for val in self.param_values]
		
	def run_first(self, game_object_control, params=[]):
		self.action = get_action(self.namespace, self.label)
		self.reinit_params(params)

	def run_not_first(self, game_object_control, params=[]):
		if self.is_real_ref:
			self.action.run(game_object_control, self.params)
		else:
			self.action.run(game_object_control, params)
			


class FireRef(Control):
	def __init__(self):
		self.param_values = []
		self.is_real_ref = True

	def reinit(self, game_object_control, params=[]):
		self.reinit_params(params)
		self.fire.reinit(game_object_control, params)

	def reinit_params(self, params):
		self.params = [val.get(params) for val in self.param_values]

	def run_first(self, game_object_control, params=[]):
		self.fire = get_fire(self.namespace, self.label)
		self.reinit_params(params)

	def run_not_first(self, game_object_control, params = []):
		if self.is_real_ref:
			self.fire.run(game_object_control, self.params)
		else:
			self.fire.run(game_object_control, params)







###########
## Values

HEUR_VALID_FORMULA = re.compile(r'^([0-9.]|\$(rand|rank|[0-9])|\(|\)|\+|-|/|\*)*$')

filter_definitions = [
  ( re.compile(r'\$rand'), '(random.random())' ),
  ( re.compile(r'\$rank'), '(RANK)'),
  ( re.compile(r'\$([0-9]+)'), r'(params[\1-1])') ]

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
	def set_formula(self, formula):
		formula.replace( '\n', '' )
		if not HEUR_VALID_FORMULA.match(formula):
			l.error( 'Invalid formula : ' + formula )
			formula='0'
		old_formula = ''
		while formula != old_formula:
			old_formula = formula
			for f in formula_filters:
				formula = f(formula)
		self.formula=formula

	def eval_formula(self, params=[]):
		try:
			return float(eval(self.formula))
		except:
			l.error( 'Invalid formula, interpreted as : ' + self.formula )
			self.formula='0'

	def get(self, params=[]):
		return self.eval_formula(params)

class BasicValue(Value):
	"""Most basic use of value.

		Exists purely to be instanciated as BasicValue( formula )
		"""
	def __init__(self, formula):
		self.set_formula(formula)
		
class Speed(Value):
	"""Has a 'type' attribute.

	Speed elements have two usages :
	- standard controls (ie. not ChangeSpeed) will simply use
	   .get_standard()
	- ChangeSpeed will have to use .get() to get a base numerical
	   value and adapt it according to .type
		"""
	def get_standard(self, game_object_control, params=[]):
		initial_value = self.get(params)
		if self.type == "absolute":
			return initial_value
		elif self.type == "relative":
			return initial_value + game_object_control.game_object.speed
		else: # sequence
			try:
				game_object_control.last_speed += self.get(params)
			except AttributeError: #.last_speed
				# default from noiz2sa
				game_object_control.last_speed = 1
			return game_object_control.last_speed

class Direction(Value):
	"""Has a 'type' attribute.

	Follows the same usage rules as Speed."""
	def get_standard(self, game_object_control, params=[]):
		initial_value = self.get(params)
		if self.type == "absolute":
			return initial_value
		elif self.type == "aim":
			return initial_value + game_object_control.game_object.aim
		elif self.type == "relative":
			return initial_value + game_object_control.game_object.direction
		else: # sequence
			try:
				game_object_control.last_direction += self.get(params)
			except AttributeError:
				game_object_control.last_direction = \
				   game_object_control.game_object.direction + self.get(params)
			return game_object_control.last_direction

			




#############
## Builders

# find appropriate name
main_actions = []
current_namespace = "pie !"

def get_random_name():
	return 'RND' + str(random.randint(100000,999999))

def get_unused_name(category):
	name = get_random_name()
	while name in namespaces[current_namespace][category]:
		name = get_random_name()
	return name

target_classes = { 
					 'bullet'          : Bullet,
					 'action'          : Action,
					 'fire'            : Fire,
					 'changeDirection' : ChangeDirection,
					 'changeSpeed'     : ChangeSpeed,
					 'accel'           : Accel,
					 'wait'            : Wait,
					 'vanish'          : Vanish,
					 'repeat'          : Repeat,
					 'bulletRef'       : BulletRef,
					 'actionRef'       : ActionRef,
					 'fireRef'         : FireRef,
					 'speed'           : Speed,
					 'direction'       : Direction,
					 }


# should I encapsulate all of this in a single object ? is there a point ?

class Builder(object):
	def __init__(self):
		# needs to have a new one per object, thus in __init__ and not in class block level
		try:
			self.target = target_classes[self.element_name]()
		except KeyError:
			pass # do nothing for classes that don't build a .target
		if self.element_name in ['actionRef', 'bulletRef', 'fireRef']:
			self.target.namespace = current_namespace

	def add_to( self, builder ):
		self.post_build()
		# $visitor calls $child.add_to_$($visitor.TYPE), passing himself as the argument on which to operate
		try:
			add_method=self.__getattribute__('add_to_' + builder.element_name )
		except:
			l.error( "Don't know what to do with %s in %s." % (self.element_name, builder.element_name) )
			return
		add_method( builder )

	def add_attrs(self, attrs):
		pass
		
	def post_build(self):
		pass

	def add_text( self, text ):
		if text:
			l.debug( "Ignoring text : " + text + " in " + self.element_name + "." )


class FormulaBuilder(Builder):
	formula = ''
	def add_text(self, text):
		# Quadratic, but it should not matter, really.
		self.formula += text

class SubActionBuilder:
	def add_to_action(self, action_builder):
		# self.post_build() has been called by Builder.add_to()
		action_builder.target.subactions.append(self.target)


class BulletmlBuilder(Builder):
	element_name =  "bulletml"


class BulletBuilder(Builder):
	element_name="bullet"

	def add_attrs(self, attrs):
		try:
			self.target.label = attrs.getValue('label')
		except KeyError:
			pass

	def add_to_fire(self, fire_builder):
		fire_builder.target.bulletref = self.get_ref()

	def get_ref(self):
		ref = BulletRef()
		ref.is_real_ref = False
		ref.namespace = current_namespace
		ref.label = self.target.label
		return ref

	def register(self):
		namespaces[current_namespace]['bullet'][self.target.label] = self.target

	def post_build(self):
		try:
			self.target.label
		except:
			self.target.label = get_unused_name('bullet')
		self.register()

	def add_to_bulletml(self, bulletml_builder):
		pass


class ActionBuilder(Builder):
	element_name="action"

	def add_attrs(self, attrs):
		try:
			self.target.label = attrs.getValue('label')
		except KeyError:
			pass

	def add_to_bulletml(self, bulletml_builder):
		if self.target.label[:3] == "top":
			namespaces[current_namespace]['main_actions'].append(self.target)

	def add_to_repeat(self, repeat_builder):
		repeat_builder.target.actionref = self.get_ref()

	def add_to_bullet(self, bullet_builder):
		bullet_builder.target.subactions.append(self.get_ref())

	def get_ref(self):
		ref = ActionRef()
		ref.is_real_ref = False
		ref.namespace = current_namespace
		ref.label = self.target.label
		# params are left untouched
		return ref
	
	def register(self):
		namespaces[current_namespace]['action'][self.target.label] = self.target

	def post_build(self):
		try:
			self.target.label
		except:
			self.target.label = get_unused_name('action')
		self.register()


class FireBuilder(Builder):
	element_name="fire"

	def add_attrs(self, attrs):
		try:
			self.target.label = attrs.getValue('label')
		except KeyError:
			pass

	def add_to_bulletml(self, bulletml_builder):
		pass

	def add_to_action(self, action_builder):
		action_builder.target.subactions.append(self.get_ref())

	def get_ref(self):
		ref = FireRef()
		ref.is_real_ref = False
		ref.namespace = current_namespace
		ref.label = self.target.label
		return ref

	def register(self):
		namespaces[current_namespace]['fire'][self.target.label] = self.target

	def post_build(self):
		try:
			self.target.label
		except:
			self.target.label = get_unused_name('fire')
		self.register()


class ChangeDirectionBuilder(Builder, SubActionBuilder):
	element_name="changeDirection"


class ChangeSpeedBuilder(Builder, SubActionBuilder):
	element_name="changeSpeed"


class AccelBuilder(Builder):
	element_name="accel"

	def build(self):
		self.target.term = self.term
		self.target.horizontal = self.horizontal
		self.target.vertical = self.vertical


class WaitBuilder(FormulaBuilder, SubActionBuilder):
	element_name="wait"

	def post_build(self):
		self.target.term_value = BasicValue(self.formula)
		


class VanishBuilder(Builder, SubActionBuilder):
	element_name="vanish"


class RepeatBuilder(Builder, SubActionBuilder):
	element_name="repeat"


class DirectionBuilder(FormulaBuilder):
	element_name="direction"
	
	def add_to_changeDirection(self, changedirection_builder):
		changedirection_builder.target.direction = self.target

	def add_to_bullet(self, bullet_builder):
		bullet_builder.target.direction = self.target

	def add_to_fire(self, fire_builder):
		fire_builder.target.direction = self.target

	def post_build(self):
		self.target.set_formula(self.formula)

	def add_attrs(self, attrs):
		try:
			self.target.type = attrs.getValue('type')
		except KeyError:
			self.target.type = "absolute"
		if self.target.type not in ['aim', 'absolute', 'relative', 'sequence']:
			self.target.type = "absolute"
			l.error("unknown direction type : " + str(self.target.type))


class SpeedBuilder(FormulaBuilder):
	element_name="speed"

	def add_to_changeSpeed(self, changespeed_builder):
		changespeed_builder.target.speed = self.target

	def add_to_bullet(self, bullet_builder):
		bullet_builder.target.speed = self.target

	def add_to_fire(self, fire_builder):
		fire_builder.target.speed = self.target

	def post_build(self):
		self.target.set_formula(self.formula)

	def add_attrs(self, attrs):
		try:
			self.target.type = attrs.getValue('type')	
		except KeyError:
			self.target.type = "absolute"
		if self.target.type not in ['absolute', 'relative', 'sequence']:
			self.target.type = "absolute"
			l.error("unknown speed type : " + str(self.target.type))


class HorizontalBuilder(FormulaBuilder):
	element_name="horizontal"

	def add_to_accel(self, accel_builder):
		accel_builder.target.horizontal_value = BasicValue( self.formula )


class VerticalBuilder(FormulaBuilder):
	element_name="vertical"

	def add_to_accel(self, accel_builder):
		accel_builder.target.vertical_value = BasicValue( self.formula )


class TermBuilder(FormulaBuilder):
	element_name="term"

	def add_to_changeDirection(self, changedirection_builder):
		changedirection_builder.target.term_value = BasicValue(self.formula)

	def add_to_changeSpeed(self, changespeed_builder):
		changespeed_builder.target.term_value = BasicValue(self.formula)

	def add_to_accel(self, accel_builder):
		accel_builder.target.term_value = BasicValue(self.formula)


class TimesBuilder(FormulaBuilder):
	element_name="times"

	def add_to_repeat(self, repeat_builder):
		repeat_builder.target.times_value = BasicValue(self.formula)


class BulletRefBuilder(Builder, SubActionBuilder):
	element_name="bulletRef"

	def add_attrs(self, attrs):
		try:
			self.target.label = attrs.getValue('label')
		except KeyError:
			pass

	def add_to_fire(self, fire_builder):
		fire_builder.target.bulletref = self.target


class ActionRefBuilder(Builder, SubActionBuilder):
	element_name="actionRef"

	def add_attrs(self, attrs):
		try:
			self.target.label = attrs.getValue('label')
		except KeyError:
			pass

	def add_to_repeat(self, repeat_builder):
		repeat_builder.target.target = self.target # clumsy


class FireRefBuilder(Builder, SubActionBuilder):
	element_name="fireRef"

	def add_attrs(self, attrs):
		try:
			self.target.label = attrs.getValue('label')
		except KeyError:
			pass

	# done


class ParamBuilder(FormulaBuilder):
	element_name="param"

	def add_to_ref(self, ref_builder):
		ref_builder.target.param_values.append( BasicValue(self.formula) )

	add_to_actionRef = add_to_ref
	add_to_fireRef = add_to_ref
	add_to_bulletRef = add_to_ref






#########################
## Parsing and Building

current_object_stack = []

builder_classes = { 
					 'bulletml'        : BulletmlBuilder,
					 'bullet'          : BulletBuilder,
					 'action'          : ActionBuilder,
					 'fire'            : FireBuilder,
					 'changeDirection' : ChangeDirectionBuilder,
					 'changeSpeed'     : ChangeSpeedBuilder,
					 'accel'           : AccelBuilder,
					 'wait'            : WaitBuilder,
					 'vanish'          : VanishBuilder,
					 'repeat'          : RepeatBuilder,
					 'direction'       : DirectionBuilder,
					 'speed'           : SpeedBuilder,
					 'horizontal'      : HorizontalBuilder,
					 'vertical'        : VerticalBuilder,
					 'term'            : TermBuilder,
					 'times'           : TimesBuilder,
					 'bulletRef'       : BulletRefBuilder,
					 'actionRef'       : ActionRefBuilder,
					 'fireRef'         : FireRefBuilder,
					 'param'           : ParamBuilder,
					 }
					 

class BulletMLHandler( xml.sax.handler.ContentHandler ):
	def characters( self, chars ):
		if current_object_stack:
			current_object = current_object_stack[-1]
			current_object.add_text( chars.strip() )

	def startDocument( self ):
		namespaces[current_namespace] = { 'action' : {},
		                                  'fire'   : {},
													 'bullet' : {},
													 'main_actions' : [] }

	def endDocument( self ):
		if not namespaces[current_namespace]['main_actions']:
			namespaces[current_namespace]['main_actions'].append(NullAction())
			l.warning( "No main action found in " + current_namespace )

	def startElement( self, name, attrs ):
		if name in builder_classes:
			builder = builder_classes[name]()
			current_object_stack.append( builder )
			builder.add_attrs(attrs) # does nothng if element doesn't like attrs
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
myBulletMLHandler = BulletMLHandler()

myParser = xml.sax.make_parser()
myParser.setFeature(xml.sax.handler.feature_validation, False)
myParser.setFeature(xml.sax.handler.feature_external_ges, False)
myParser.setContentHandler(myBulletMLHandler)

def set_action_namespace( name ):
	global current_namespace
	current_namespace = name

def get_main_actions( name ):
	if not name in namespaces:
		set_action_namespace( name )
		try:
			f = open( name, 'r' )
			myParser.parse(f)
			f.close()
		except Exception,ex:
			l.error( "Error while parsing BulletML file : " + name )
			l.debug("Exception :" + str(ex))
			raise
			return namespaces["null"]['main_actions']
	return copy.deepcopy(namespaces[name]['main_actions'])

class GameObjectSubController:
	def run(self, params=[]):
		self.top_action.run(self, params)
		# "top_action" sucks

class GameObjectMainController:
	def __init__(self):
		self.sub_controllers = []
		self.params = []

	def add_action(self, action):
		sub_controller = GameObjectSubController()
		sub_controller.top_action = action
		sub_controller.game_object = self.game_object
		self.sub_controllers.append(sub_controller)

	def set_behavior( self, name ): # name is really a namepace
		master_actions = get_main_actions(name)
		self.sub_controllers = []
		for master_action in master_actions:
			self.add_action(master_action)

	def set_game_object(self, game_object):
		self.game_object = game_object
		for sub_controller in self.sub_controllers:
			sub_controller.game_object = game_object

	def run(self):
		for sub_controller in self.sub_controllers:
			sub_controller.run(self.params)


############
## Testing
##  hopelessly bit rotten

class FakeGameObject:
	def __init__(self):
		self.control = GameObjectController()
		self.control.game_object = self
		self.control.set_behavior('bee.xml')

	def fire(self, bullet_control):
		print "launching ", bullet_control, " in hyperspace"

	direction = 0.0
	speed = 1.0

if __name__ == '__main__':
	gamobj = FakeGameObject()
	for i in xrange(500):
		#print get_action("test.xml","topmove").subactions
		gamobj.control.run()
		#print (gamobj.direction, gamobj.speed)
		#print i
