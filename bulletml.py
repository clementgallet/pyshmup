# -*- coding: utf-8 -*-

import random
import logging
import re
import xml.sax, xml.sax.handler
import math

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




#######################
# BulletML constructs

# This contains all the bulletml elements pertaining to a file (namespace)
# namespaces[namespace]['action'|'fire'|'bullet'][label] = <controller_object>
# namespaces[namespace]['main_actions'] = [top_action1, top_action2, ...]
namespaces = { "null" : { 'action' : {}, 'fire' : {}, 'bullet' : {}, 'main_actions' : [] } }

def get_action(namespace, label):
	return namespaces[namespace]['action'][label]

def get_fire(namespace, label):
	return namespaces[namespace]['fire'][label]

def get_bullet(namespace, label):
	return namespaces[namespace]['bullet'][label]




###################################
## Controller objects
## (reflecting BulletML elements)

# run() status codes
NOT_DONE = 1 # action is not yet finished
DONE = 2     # action is finished
# those are communicated via game_object_control.turn_status

class Cookie:
	"""Simple data class to hold the state of an element and its children."""
	def __init__(self, parent_cookie = None):
		self.subcookies = {}
		self.new = True
		if parent_cookie is not None:
			self.values = parent_cookie.values
		else:
			self.values = []

class Control:
	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			l.warning(str(self) + "does not implement run.")
		

class Bullet(Control):
	"""A Bullet is merely a bunch of actions.
	
	Not runnable per se."""
	def __init__(self):
		self.subactions = []


# if an action is found in a file, build the action separately and ref it

class Action(Control):
	"""An action contains other sub-actions and runs them in order,
	waiting for one to signal completion before getting to the next one."""
	def __init__(self):
		self.subactions = []

	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			for subaction in self.subactions:
				if subaction not in cookie.subcookies:
					cookie.subcookies[subaction] = Cookie(cookie)
				else:
					cookie.subcookies[subaction].new = True
			cookie.current_index = 0

		if cookie.current_index >= self.sub_length:
			game_object_control.turn_status = DONE
		else:
			self.subactions[cookie.current_index].run(game_object_control, \
			        cookie.subcookies[self.subactions[cookie.current_index]])
			if game_object_control.turn_status == DONE:
				cookie.current_index += 1
				for subcookie in cookie.subcookies:
					subcookie.new = True
				self.run(game_object_control, cookie)
			if cookie.current_index >= self.sub_length:
				game_object_control.turn_status = DONE
			else:
				game_object_control.turn_status = NOT_DONE


class Fire(Control):
	"""Gets a bullet from its bulletref and calls the game_object's fire() method,
	passing it an appropriate cookie."""
	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False

			if self.bulletref not in cookie.subcookies:
				cookie.subcookies[self.bulletref] = Cookie(cookie)

			main_control = BulletMLController()
			main_control.game_object = game_object_control.game_object

			child_cookie = Cookie()
			bref_cookie = cookie.subcookies[self.bulletref]

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

			# where do the params come from ?
			NOWHERE = 0
			BULLET = 1
			FIRE = 2

			speed_location = NOWHERE
			direction_location = NOWHERE

			try:
				speed = bullet.speed
				speed_location = BULLET
			except AttributeError: pass
			try:
				direction = bullet.direction
				direction_location = BULLET
			except AttributeError: pass

			try:
				speed = self.speed
				speed_location = FIRE
			except AttributeError: pass
			try:
				direction = self.direction
				direction_location = FIRE
			except AttributeError: pass


			if speed_location == FIRE:
				numeric_speed = speed.get_standard(game_object_control, cookie)
			elif speed_location == BULLET:
				numeric_speed = speed.get_standard(game_object_control, bref_cookie)
			else:
				numeric_speed = game_object_control.game_object.speed
				
			if direction_location == FIRE:
				numeric_direction = direction.get_standard(game_object_control, cookie)
			elif direction_location == BULLET:
				numeric_direction = direction.get_standard(game_object_control, bref_cookie)
			else:
				numeric_direction = game_object_control.game_object.direction
				if self.is_horizontal:
					numeric_direction -= 90


			if self.bulletref.is_real_ref:
				child_cookie.values = self.bulletref.get_values(bref_cookie)
			else:
				child_cookie.values = cookie.values

			main_control.cookie = child_cookie

			if bullet.subactions:
				main_control.game_object.fireml(main_control, numeric_direction, numeric_speed)
			else:
				main_control.game_object.firenoml(numeric_direction, numeric_speed)


			game_object_control.last_speed = numeric_speed
			game_object_control.last_direction = numeric_direction

		game_object_control.turn_status = DONE


class ChangeDirection(Control):
	# has two values :
	#  term and direction

	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			cookie.initial_direction = game_object_control.game_object.direction
			cookie.direction_offset = self.direction.get_standard(game_object_control, cookie) - \
											cookie.initial_direction
			cookie.direction_offset = cookie.direction_offset % 360
			if cookie.direction_offset > 180:
				cookie.direction_offset -= 360
			elif cookie.direction_offset < 180:
				cookie.direction_offset += 360
			cookie.frame = 0
			cookie.term = self.term_value.get(cookie)
		
		if cookie.frame >= cookie.term:
			game_object_control.turn_status =  DONE
		else:
			cookie.frame += 1
			game_object_control.game_object.direction = cookie.initial_direction + \
			   (float(cookie.frame) / cookie.term) * cookie.direction_offset
			game_object_control.turn_status =  NOT_DONE


class ChangeSpeed(Control):
	# has two values :
	#  term and speed

	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			initial_speed = game_object_control.game_object.speed
			cookie.frame = 0
			cookie.term = int(self.term_value.get(cookie))
			if self.speed.type == "absolute":
				cookie.theoretical_speed = initial_speed
				cookie.speed_offset = float(self.speed.get(cookie) - \
							    initial_speed) / cookie.term
				cookie.get_new_speed = self.get_new_speed_absolute
			elif self.speed.type == "relative":
				cookie.theoretical_speed = initial_speed
				cookie.speed_offset = float(self.speed.get(cookie)) / cookie.term
				cookie.get_new_speed = self.get_new_speed_relative
			else: # "sequence"
				cookie.last_speed = game_object_control.game_object.speed
				cookie.get_new_speed = self.get_new_speed_sequence
		
		if cookie.frame >= cookie.term:
			game_object_control.turn_status = DONE
		else:
			cookie.frame += 1
			game_object_control.game_object.speed = \
			       cookie.get_new_speed(game_object_control, cookie)
			game_object_control.turn_status = NOT_DONE


	def get_new_speed_absolute(self, game_object_control, cookie):
		cookie.theoretical_speed += cookie.speed_offset
		return cookie.theoretical_speed

	def get_new_speed_relative(self, game_object_control, cookie):
		cookie.theoretical_speed += cookie.speed_offset
		return cookie.theoretical_speed
		

	def get_new_speed_sequence(self, game_object_control, params=[]):
		game_object_control.last_speed += self.speed.get(cookie)
		return game_object_control.last_speed



class Accel(Control):
	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			cookie.term = int(self.term_value.get(cookie))
			cookie.frame = 0

			try:
				cookie.horiz = self.horiz_value.get(cookie)
			except AttributeError:
				cookie.horiz = 0
			if cookie.term > 0:
				cookie.horiz /= cookie.term
				
			try:
				cookie.vert = self.vert_value.get(cookie)
			except AttributeError:
				cookie.vert = 0
			if cookie.term > 0:
				cookie.vert /= cookie.term

		if cookie.frame > cookie.term:
			game_object_control.turn_status = DONE
		else:
			cookie.frame += 1
			game_object_control.turn_status = NOT_DONE
			if not (cookie.horiz or cookie.vert):
				return
			initial_speed = game_object_control.game_object.speed
			initial_direction = game_object_control.game_object.direction
			# on se ramene au cas vertical
			if self.is_horizontal:
				initial_direction -= 90
				dy = cookie.horiz
				dx = cookie.vert
			else:
				dx = cookie.horiz
				dy = - cookie.vert
			xx =   initial_speed * math.sin( initial_direction * math.pi / 180 )
			yy = - initial_speed * math.cos( initial_direction * math.pi / 180 )
			xx += dx
			yy += dy
			if abs(yy) < 0.000001:
				if xx>0:
					direction = 90
					speed = xx
				else:
					direction = -90
					speed = -xx
			else:
				direction = math.atan( - xx / yy ) * 180 / math.pi
				if yy>0:
					direction += 180
				speed = math.sqrt( xx*xx + yy*yy )
			if self.is_horizontal:
				direction += 90
			game_object_control.game_object.speed = speed
			game_object_control.game_object.direction = direction

					

class Wait(Control):
	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			game_object_control.game_object.wait = self.term_value.get(cookie)
			game_object_control.turn_status = NOT_DONE
		else:	
			game_object_control.turn_status = DONE


class Vanish(Control):
	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			game_object_control.game_object.vanish()
		else:
			l.warning("Same bullet vanishing several times.")



class Repeat(Control):
	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			cookie.times = self.times_value.get(cookie)
			cookie.repetition = 0
			cookie.subcookies[self.actionref] = Cookie(cookie)

		if cookie.repetition > cookie.times:
			game_object_control.turn_status = DONE
		else:
			self.actionref.run(game_object_control, cookie.subcookies[self.actionref])
			if game_object_control.turn_status == DONE:
				cookie.repetition += 1
				if cookie.repetition != cookie.times:
					cookie.subcookies[self.actionref].new = True
					self.run(game_object_control, cookie)
			game_object_control.turn_status = NOT_DONE



class BulletRef(Control):
	def __init__(self):
		self.param_values = []
		self.is_real_ref = True

	def get_values(self, cookie):
		return [val.get(cookie) for val in self.param_values]

	def get_bullet(self):
		return get_bullet(self.namespace, self.label)



class ActionRef(Control):
	def __init__(self):
		self.param_values = []
		self.is_real_ref = True

	def get_values(self, cookie):
		return [val.get(cookie) for val in self.param_values]
		
	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			self.action = get_action(self.namespace, self.label)
			if self.action not in cookie.subcookies:
				cookie.subcookies[self.action] = Cookie(cookie)
			cookie.subcookies[self.action].new = True

			if self.is_real_ref:
				cookie.subcookies[self.action].values = self.get_values(cookie)

		self.action.run(game_object_control, cookie.subcookies[self.action])
			


class FireRef(Control):
	def __init__(self):
		self.param_values = []
		self.is_real_ref = True

	def get_values(self, cookie):
		return [val.get(cookie) for val in self.param_values]

	def run(self, game_object_control, cookie):
		if cookie.new:
			cookie.new = False
			self.fire = get_fire(self.namespace, self.label)
			if self.fire not in cookie.subcookies:
				cookie.subcookies[self.fire] = Cookie(cookie)
			cookie.subcookies[self.fire].new = True

			if self.is_real_ref:
				cookie.subcookies[self.fire].values = self.get_values(cookie)

		self.fire.run(game_object_control, cookie.subcookies[self.fire])







###########
## Values

HEUR_VALID_FORMULA = re.compile(r'^([0-9.]|\$(rand|rank|[0-9])|\(|\)|\+|-|/|\*)*$')

filter_definitions = [
  ( re.compile(r'\$rand'), '(random.random())' ),
  ( re.compile(r'\$rank'), '(RANK)'),
  ( re.compile(r'\$([0-9]+)'), r'(cookie.values[\1-1])') ]

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
		# Non-(harmfulness potential) checking
		if not HEUR_VALID_FORMULA.match(formula):
			l.error('Invalid formula : ' + formula)
			formula='0'
		# Performing substitutions for future eval()uation
		old_formula = ''
		while formula != old_formula:
			old_formula = formula
			for f in formula_filters:
				formula = f(formula)
		self.formula=formula

	def eval_formula(self, cookie):
		try:
			return float(eval(self.formula))
		except:
			l.error('Invalid formula, interpreted as : ' + self.formula)
			self.formula='0'
			return 0

	def get(self, cookie):
		return self.eval_formula(cookie)

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
	def get_standard(self, game_object_control, cookie):
		initial_value = self.get(cookie)
		if self.type == "absolute":
			return initial_value
		elif self.type == "relative":
			return initial_value + game_object_control.game_object.speed
		else: # sequence
			try:
				game_object_control.last_speed += self.get(cookie)
			except AttributeError: #.last_speed
				# default from noiz2sa
				game_object_control.last_speed = 1
			return game_object_control.last_speed

class Direction(Value):
	"""Has a 'type' attribute.

	Follows the same usage rules as Speed."""
	def get_standard(self, game_object_control, cookie):
		initial_value = self.get(cookie)
		if self.type == "absolute":
			numeric_direction =  initial_value
			if self.is_horizontal:
				numeric_direction -= 90
		elif self.type == "aim":
			if game_object_control.game_object.aimed_player is not None:
				delta_x = game_object_control.game_object.aimed_player.x - game_object_control.game_object.x
				delta_y = game_object_control.game_object.aimed_player.y - game_object_control.game_object.y
				if abs(delta_y) < 0.000001:
					if (delta_x) > 0:
						aim = 90
					else:
						aim = -90
				else:
					aim = math.atan(- delta_x / delta_y) * 180 / math.pi
					if delta_y > 0:
						aim += 180
			else:
				aim = 0
			numeric_direction = initial_value + aim
		elif self.type == "relative":
			numeric_direction = initial_value + game_object_control.game_object.direction
		else: # sequence
			try:
				game_object_control.last_direction += self.get(cookie)
			except AttributeError:
				game_object_control.last_direction = \
				   game_object_control.game_object.direction + self.get(cookie)
			numeric_direction = game_object_control.last_direction
		return numeric_direction

			



#############
## Builders

# find appropriate name
main_actions = []
current_namespace = "pie !"
is_horizontal = False

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

# Common builders infrastructure, including magic .add_to()
class Builder(object):
	def __init__(self):
		# needs to have a new one per object, thus in __init__ and not in class block level
		try:
			self.target = target_classes[self.element_name]()
		except KeyError:
			pass # do nothing for classes that don't build a .target
		if self.element_name in ['actionRef', 'bulletRef', 'fireRef']:
			self.target.namespace = current_namespace

	def add_to(self, builder):
		self.post_build()
		# $parent calls $child.add_to_$($parent.TYPE), passing himself as the argument on which to operate
		# semi-visitor pattern
		try:
			add_method=self.__getattribute__('add_to_' + builder.element_name)
		except:
			l.error("Don't know what to do with %s in %s." % (self.element_name, builder.element_name))
			return
		add_method(builder)

	def add_attrs(self, attrs):
		pass
		
	def post_build(self):
		pass

	def add_text( self, text ):
		if text:
			l.debug( "Ignoring text : " + text + " in " + self.element_name + "." )


# Basic text aggregator
class FormulaBuilder(Builder):
	formula = ''
	def add_text(self, text):
		# Quadratic, but it should not matter, really.
		self.formula += text

# Builders of elements that can be subactions of another element
# should inherit from this as well as Builder
class SubActionBuilder:
	def add_to_action(self, action_builder):
		# self.post_build() has been called by Builder.add_to()
		action_builder.target.subactions.append(self.target)

# Top-level element, not actually reflected but its attributes are
# passed down by build-time global variables
class BulletmlBuilder(Builder):
	element_name =  "bulletml"

	def add_attrs(self, attrs):
		global is_horizontal
		try:
			type = attrs.getValue('type')
		except KeyError:
			type = "none"
		if type == "horizontal":
			is_horizontal = True
		else:
			is_horizontal = False


class BulletBuilder(Builder):
	element_name="bullet"

	def add_attrs(self, attrs):
		try:
			self.target.label = attrs.getValue('label')
		except KeyError:
			self.target.label = get_unused_name('bullet')

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
		self.target.sub_length = len(self.target.subactions)
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
		self.target.is_horizontal = is_horizontal


class ChangeDirectionBuilder(Builder, SubActionBuilder):
	element_name="changeDirection"


class ChangeSpeedBuilder(Builder, SubActionBuilder):
	element_name="changeSpeed"


class AccelBuilder(Builder, SubActionBuilder):
	element_name="accel"

	def post_build(self):
		self.target.is_horizontal = is_horizontal

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
		self.target.is_horizontal = is_horizontal

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
		accel_builder.target.horiz_value = BasicValue( self.formula )


class VerticalBuilder(FormulaBuilder):
	element_name="vertical"

	def add_to_accel(self, accel_builder):
		accel_builder.target.vert_value = BasicValue( self.formula )


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


class ParamBuilder(FormulaBuilder):
	element_name="param"

	def add_to_ref(self, ref_builder):
		ref_builder.target.param_values.append(BasicValue(self.formula))

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
					 
# Sax events handler
class BulletMLHandler(xml.sax.handler.ContentHandler):
	def characters(self, chars):
		if current_object_stack:
			current_object = current_object_stack[-1]
			current_object.add_text(chars.strip())

	# Create a new namespace for the document
	def startDocument(self):
		namespaces[current_namespace] = { 'action' : {},
		                                  'fire'   : {},
													 'bullet' : {},
													 'main_actions' : [] }

	# Eventually add NullAction if no top action is found
	def endDocument(self):
		if not namespaces[current_namespace]['main_actions']:
			l.warning("No main action found in " + current_namespace)

	# Create builders mirroring xml contents
	def startElement(self, name, attrs):
		if name in builder_classes:
			builder = builder_classes[name]()
			current_object_stack.append(builder)
			builder.add_attrs(attrs) # does nothing if element doesn't like attrs
		else:
			l.warning("Unknown element : " + name)

	def startElementNS(self, name, qname, attrs):
		uri, localname = name
		if uri == "http://www.asahi-net.or.jp/~cs8k-cyu/bulletml":
			self.startElement(localname, attrs)
		else:	
			l.warning("Unknown element : " + qname)

	# Add builder to parent
	def endElement(self, name):
		if name in builder_classes:
			try:
				builder = current_object_stack.pop()
				if current_object_stack:
					parent_builder = current_object_stack[-1]
					builder.add_to(parent_builder)
			except:
				raise
	
	def endElementNS(self, name, qname):
		uri, localname = name
		if uri == "http://www.asahi-net.or.jp/~cs8k-cyu/bulletml":
			self.endElement(name)

#FIXME: find a slightly less moronic name
myBulletMLHandler = BulletMLHandler()

myParser = xml.sax.make_parser()
myParser.setFeature(xml.sax.handler.feature_validation, False)
myParser.setFeature(xml.sax.handler.feature_external_ges, False)
myParser.setContentHandler(myBulletMLHandler)

def get_main_actions(name):
	if not name in namespaces:
		global current_namespace
		current_namespace = name
		try:
			f = open(name, 'r')
			myParser.parse(f)
			f.close()
		except Exception,ex:
			l.error("Error while parsing BulletML file : " + str(name))
			l.debug("Exception :" + str(ex))
			raise
			return namespaces["null"]['main_actions']
	return namespaces[name]['main_actions']



##########################
## Top-level controllers

# Contain top actions of a BulletMLController
# Necessary to allow for separate params !?
# FIXME: statute on this class's right to live
class BulletMLSubController:
	def run(self, cookie):
		self.top_action.run(self, cookie)


# Exported abstract controller
class BulletMLController:
	def __init__(self):
		self.sub_controllers = []
		self.cookie = Cookie()

	def add_action(self, action):
		sub_controller = BulletMLSubController()
		sub_controller.top_action = action
		sub_controller.game_object = self.game_object
		self.sub_controllers.append(sub_controller)

	def set_behavior(self, name): # name is really a namepace
		master_actions = get_main_actions(name)
		self.sub_controllers = []
		for master_action in master_actions:
			self.add_action(master_action)

	def set_game_object(self, game_object):
		self.game_object = game_object
		for sub_controller in self.sub_controllers:
			sub_controller.game_object = game_object

	def run(self):
		if self.cookie.new:
			self.cookie.new = False
			for sub_controller in self.sub_controllers:
				self.cookie.subcookies[sub_controller] = Cookie(self.cookie)
		for sub_controller in self.sub_controllers:
			sub_controller.run(self.cookie.subcookies[sub_controller])
