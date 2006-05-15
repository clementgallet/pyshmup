# -*- coding: utf-8 -*-

import bulletml
from bulletml import BulletMLController
import sprite

NO_GRAPHICS = False

import pygame
import logging
import math
import time
import random
import copy
from OpenGL.GL import * # evil
from OpenGL.GLU import * 

#import profile

#############
## Constants

KEY_SHOT = pygame.K_q
#KEY_SHOT = pygame.K_W

DRAW_HITBOX = True

FONKY_LINES = True

WIDTH = 640
HEIGHT = 480

SHIP_BITMAP = "data/images/ship.png"
#SHIP_BITMAP = "data/images/shot_small.png"
BULLET_BITMAP = "data/images/shot.png"
#BULLET_BITMAP = "data/images/shot_small.png"
FOE_BITMAP = "data/images/foe.png"
#FILE = "data/bullets/doud.xml"
#FILE = "data/bullets/doud_synch.xml"
#FILE = "data/bullets/doud_directed_rand_smooth.xml"
#FILE = "data/bullets/doud_circles.xml"
#FILE = 'data/bullets/struggle.xml'
#FILE = 'data/bullets/bee.xml'
#FILE = 'data/bullets/slowdown.xml'
#FILE = 'data/bullets/beewinder.xml'
#FILE = 'data/bullets/side_cracker.xml'
#FILE = 'data/bullets/roll3pos.xml'
#FILE = 'data/bullets/rollbar.xml'
FILE = 'data/bullets/keylie_360.xml'
#FILE = 'data/bullets/double_roll_seeds.xml'
#FILE = 'data/bullets/[ketsui]_r4_boss_rb_rockets.xml'
#FILE = 'data/bullets/quad3.xml'
#FILE = 'data/bullets/roll.xml'
#FILE = 'data/bullets/4waccel.xml'
#FILE = 'data/bullets/248shot.xml'
#FILE = 'data/bullets/bar.xml'
#FILE = 'data/bullets/[Ikaruga]_r5_vrp.xml'
#FILE = 'data/bullets/[Progear]_round_3_boss_back_burst.xml'

OUT_LIMIT = 0.2 # out-of-screen is e.g. x > width*(1+OUT_LIMIT)

RADIUS = 3.0 # player's "hit-disc" radius, in game units

PLAYER_SPEED = 2.0

SCALE = 400 # screen height in game units

RANK = 0.5 # difficulty setting, 0 to 1

FPS = 60
MAX_SKIP = 9

SHOT_TIME = 40
SHOT_FAST = 1 # in presses every SHOT_TIME frames

BACKGROUND_COLOR = (.235, .275, .275, 1)

#####################
## Derived constants

RADIUS2 = RADIUS * RADIUS

UNIT_HEIGHT = SCALE/2
UNIT_WIDTH = (UNIT_HEIGHT * WIDTH) / HEIGHT

if FONKY_LINES:
	LINE_RADIUS2 = 10000*RADIUS2
	FONKY_COLOR = [.205, .245, .245, 1]
	FONKY_OFFSET = [ BACKGROUND_COLOR[i] - FONKY_COLOR[i] for i in [0,1,2] ]

############
## Logging

console = logging.StreamHandler( )
console.setLevel( logging.DEBUG )
formatter = logging.Formatter( '%(name)-12s: %(levelname)-8s %(message)s' )
console.setFormatter( formatter )
logging.getLogger('').addHandler( console )

l = logging.getLogger('main')
l.setLevel(logging.DEBUG)

############
## Graphics

def set_perspective(width, height):
	global screen
	screen = pygame.display.set_mode( (width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE )

	glViewport(0, 0, width, height)

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	fov = 30
	dist = SCALE*math.tan(math.pi/8)/math.tan(fov*math.pi/360)
#	gluPerspective(fov, float(width)/height, dist*0.5, dist*1.5)
	gluPerspective(fov, float(WIDTH)/HEIGHT, dist*0.5, dist*1.5)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	glTranslate(0, 0, -dist)#

images = {}

bulletml.RANK = RANK

def get_image(name):
	if name in images:
		return images[name]
	else:
		try:
			image = pygame.image.load(name)
			images[name] = image
		except:
			l.error("Failed to load image : " + str(name))
			images[name] = images['NULL']
		return images[name]

images['NULL'] = pygame.image.load('data/images/not_found.png')
			
		
def init_sdl():
	if not pygame.image.get_extended():
		l.error("No SDL_Image support. Aborting.")
		return False

	passed, failed = pygame.init()
	if failed > 0:
		count = 0
		try: pygame.display.init()
		except Exception, ex:
			l.error("Display failed to init : " + str(ex))
			count += 1
			return False
		try: pygame.mixer.init()
		except Exception, ex:
			l.error("Mixer failed to init : " + str(ex))
			l.error("Disabling sound system.")
			count += 1
		if count < failed:
			l.warning("Some SDL modules failed to initialize.")

	global screen
	screen = pygame.display.set_mode( (WIDTH,HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE )

	set_perspective(WIDTH, HEIGHT)

	glClearColor(*BACKGROUND_COLOR)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_BLEND)

	glEnable(GL_TEXTURE_2D)

	pygame.display.set_caption("FIVE TONS OF FLAX !")
	
		
																						
# this is not intended to be particularly exten{ded,sible}
update_list = []
bullet_list = []
player_list = []
foe_list = []

class Foe(object):
	def __init__(self):
		self.x = UNIT_WIDTH*0
		self.y = UNIT_HEIGHT*0.3
		self.direction = 0
		self.speed = 0

		self.sprite = sprite.get_sprite( FOE_BITMAP )

		update_list.append(self)
		foe_list.append(self)
		self.to_remove = False
		
	def draw(self):
		glPushMatrix()
		# TODO: if foe has no target, it becomes transparent
		glColor4f(1.0, 1.0, 1.0, 1.0)
		glTranslatef(self.x, self.y, 0)
		self.sprite.draw()
		glPopMatrix()


	def update(self):
		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		   self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT):
			self.to_remove = True
		self.x += math.sin(self.direction*math.pi/180)*self.speed
		self.y -= math.cos(self.direction*math.pi/180)*self.speed

	def fire(self, direction=None, speed=None, new_bullet=None):
		if new_bullet is None:
			new_bullet = SimpleBullet()
		new_bullet.x = self.x
		new_bullet.y = self.y
		if direction is not None:
			new_bullet.direction = direction
		else:
			new_bullet.direction = self.direction
		if speed is not None:
			new_bullet.speed = speed
		else:
			new_bullet.speed = self.speed

class BulletMLFoe(Foe):
	#FIXME: refactor this and SimpleBulletML
	def __init__(self, bulletml_behav):
		self.aim = 0
		self.delta_x = 0
		self.delta_y = 0
		self.aimed_player = None
		self.controller = BulletMLController()
		self.controller.game_object = self
		self.controller.set_behavior(bulletml_behav)
		super(BulletMLFoe, self).__init__()

	def fire(self, controller, direction=None, speed=None):
		if not controller.sub_controllers:
			new_bullet = SimpleBullet()
		else:
			new_bullet = SimpleBulletML()
			new_bullet.controller = controller
			controller.set_game_object(new_bullet)
		super(BulletMLFoe, self).fire(direction, speed, new_bullet)

	def update(self):
		#FIXME: the first player is always aimed, ahahah !!
		self.aimed_player = player_list[0]
		self.delta_x = self.aimed_player.x - self.x
		self.delta_y = self.aimed_player.y - self.y
		if abs(self.delta_y) < 0.000001:
			if (self.delta_x) > 0:
				self.aim = 90
			else:
				self.aim = -90
		else:
			self.aim = math.atan(- self.delta_x / self.delta_y) * 180 / math.pi
			if self.delta_y > 0:
				self.aim += 180
		self.controller.run()
		super(BulletMLFoe, self).update()

class SimpleBullet(object):
	def __init__(self):
		self.direction = 0
		self.speed = 0
		self.x = 0
		self.y = UNIT_HEIGHT * 0.5

		self.sprite = sprite.get_sprite( BULLET_BITMAP )

		update_list.append(self)
		bullet_list.append(self)
		self.to_remove = False

		self.t = 0
		self.sin_spd = random.random() * 0.04
		
	def fire(self, direction=None, speed=None, new_bullet=None):
		if new_bullet is None:
			new_bullet = SimpleBullet()
		new_bullet.x = self.x
		new_bullet.y = self.y
		if direction is not None:
			new_bullet.direction = direction
		else:
			new_bullet.direction = self.direction
		if speed is not None:
			new_bullet.speed = speed
		else:
			new_bullet.speed = self.speed
			
	def update(self):
		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		   self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT):
			self.to_remove = True
		self.x += math.sin(self.direction*math.pi/180)*self.speed
		self.y -= math.cos(self.direction*math.pi/180)*self.speed
		self.t = (self.t+self.sin_spd) % (2*math.pi)

	def draw(self):
		glPushMatrix()
		glColor4f(1.0, 1.0, 1.0, 0.2)
		glTranslatef(self.x, self.y, 0)#math.sin(self.t)*5)
#		glRotatef(self.t * 180/math.pi, 0, 0, 1)
		self.sprite.draw()
		glPopMatrix()
		#glTranslatef(-self.x, -self.y, 0)

	def vanish(self):
		self.to_remove = True


class SimpleBulletML(SimpleBullet):
	def __init__(self, bulletml_behav=None):
		self.aim = 0
		self.delta_x = 0
		self.delta_y = 0
		self.aimed_player = None
		self.controller = BulletMLController()
		self.controller.game_object = self
		if bulletml_behav is not None:
			self.controller.set_behavior(bulletml_behav)
		super(SimpleBulletML, self).__init__()

	def fire(self, controller, direction=None, speed=None):
		if not controller.sub_controllers:
			new_bullet = SimpleBullet()
		else:
			new_bullet = SimpleBulletML()
			new_bullet.controller = controller
			controller.set_game_object(new_bullet)
		super(SimpleBulletML, self).fire(direction, speed, new_bullet)

	def update(self):
		self.aimed_player = player_list[0]
		self.delta_x = self.aimed_player.x - self.x
		self.delta_y = self.aimed_player.y - self.y
		if abs(self.delta_y) < 0.000001:
			if (self.delta_x) > 0:
				self.aim = 90
			else:
				self.aim = -90
		else:
			self.aim = math.atan(- self.delta_x / self.delta_y) * 180 / math.pi
			if self.delta_y > 0:
				self.aim += 180
		self.controller.run()
		super(SimpleBulletML, self).update()

		
		

SHOT_NO = 0
SHOT_LOW = 1
SHOT_HIGH = 5

class Player:
	def update(self):
		keys = pygame.key.get_pressed()
		dx = 0
		dy = 0
		if keys[pygame.K_RIGHT]:
			dx += 1
		if keys[pygame.K_LEFT]:
			dx -= 1
		if keys[pygame.K_UP]:
			dy += 1
		if keys[pygame.K_DOWN]:
			dy -= 1
		self.x += dx*PLAYER_SPEED
		self.y += dy*PLAYER_SPEED
		#s = ([[b.x,b.y,self.x, self.y, (b.x-self.x)**2+(b.x-self.y)**2] for b in bullet_list])# < RADIUS:
		#random.shuffle(s)
		#print s[0]
		if bullet_list:
			if min([(b.x-self.x)**2+(b.y-self.y)**2 for b in bullet_list]) < RADIUS2:
				self.to_remove = True
			
		# "front montant"
		shot_pressed = (not self.last_shot_pressed) and keys[KEY_SHOT]
		self.last_shot_pressed = keys[KEY_SHOT]
		
		# find shot_state
		if self.shot_state == 0:
			if shot_pressed:
				self.to_next_shot_limit = SHOT_TIME
				self.shot_count = 0
				self.last_shot_state = 0
				self.shot_state = 1
		else:
			if self.to_next_shot_limit <= 0:
				# we have to decide the next firepower
				if self.shot_count >= self.shot_state:
					self.last_shot_state = self.shot_state
					if self.shot_state < SHOT_HIGH:
						self.shot_state += 1
				elif self.shot_count > 0:
					# statu quo
					self.last_shot_state = self.shot_state
				else:
					if self.last_shot_state >= self.shot_state:
						self.last_shot_state = self.shot_state
						if self.shot_state > 1:
							self.shot_state -= 1
						else:
							if self.shot_count > 0:
								self.last_shot_state = 1 # staying in LOW anyway
							else:
								self.shot_state = SHOT_NO
					else:
						self.last_shot_state = self.shot_state
				self.to_next_shot_limit = SHOT_TIME
				self.shot_count = 0
				if shot_pressed:
					self.shot_count = 1
			else:
				if shot_pressed:
					self.shot_count += 1
				self.to_next_shot_limit -= 1

		#print "state ", self.shot_state, " for ", self.to_next_shot_limit, " (", self.shot_count, ") [", \
		#	shot_pressed, "]"
		global BACKGROUND_COLOR
#		BACKGROUND_COLOR = ( 50* self.shot_state, 50* self.shot_state, 50* self.shot_state )

		#TODO: show it !
					
				
			

	def __init__(self):
		self.x = 0.0
		self.y = -UNIT_HEIGHT * .5
		self.to_remove = False
		update_list.append(self)
		player_list.append(self)

		self.shot_state = SHOT_NO
		self.to_next_shot_limit = 0
		self.last_shot_pressed = False
		self.shot_pressed = False
		self.shot_count = 0

		self.sprite = sprite.get_sprite(SHIP_BITMAP)

		i = 1
		while glIsList(i):
			i += 1
		glNewList(i, GL_COMPILE)
		
		glColor4f(1.0, 0.0, 1.0, 0.5)
		glDisable(GL_TEXTURE_2D)
		glBegin(GL_TRIANGLE_FAN)
		NB_STRIPS = 32
		glVertex2f(0., 0.)
		for k in xrange(NB_STRIPS+1):
			glVertex2f(RADIUS * math.cos(2 * math.pi * k / NB_STRIPS),
			           RADIUS * math.sin(2 * math.pi * (-k) / NB_STRIPS))
		glEnd()
		glColor4f(1., 1., 1., 1.)

		glEndList()
		self.circle_list = i

		self.t=0
		
	def draw(self):
		if FONKY_LINES:
			for b in bullet_list:
				coeff = ((float(self.x)-b.x)**2+(self.y-b.y)**2)/LINE_RADIUS2
				if coeff <= 1:
					FONKY_COLOR[3] = (1-coeff) ** 2 # alpha component
					glColor4f(*FONKY_COLOR)
					glBegin(GL_LINES)
					glVertex2f(b.x, b.y)
					glVertex2f(self.x, self.y)
					glEnd()
			glColor4f(1.0, 1.0, 1.0, 1.0)
		#if DRAW_HITBOX and FONKY_LINES:
		#	pygame.draw.circle(screen, FONKY_COLOR, (int(self.x), int(self.y)), int(RADIUS))
		glPushMatrix()
		glTranslatef(self.x, self.y, 0)
		self.t = (self.t+1)%360
		glRotatef(self.t, 0, 0, 1)
		self.sprite.draw()
		glCallList(self.circle_list)
		glPopMatrix()
		

fskip = 0
next_ticks = 0
def get_turn_actions():
	global next_ticks
	global skip_c, fskip
	if next_ticks == 0:
		next_ticks = pygame.time.get_ticks() + 1000/FPS
	new_ticks = pygame.time.get_ticks()
	if new_ticks <= next_ticks:
		pygame.time.wait(next_ticks - new_ticks)
		fskip = 0
		next_ticks += 1000/FPS
		return 2
	elif fskip > MAX_SKIP:
		next_ticks = new_ticks + 1000/FPS
		fskip = 0
		return 2
	else:
		skip_c += 1
		fskip += 1
		next_ticks += 1000/FPS
		return 1

skip_c = 0
def main():
	global bullet_list
	init_sdl()
	x=0
	first_ticks = pygame.time.get_ticks()
	frame = 0
	vframe = 0
	max_ob = 0
	Player()
#	SimpleBulletML(FILE)
#	Foe()
	BulletMLFoe(FILE)
	while bullet_list or foe_list:
#	while pygame.time.get_ticks() < 80000:
		for ev in pygame.event.get():
			if ev.type == pygame.VIDEORESIZE:
				set_perspective(ev.w, ev.h)
			if ev.type == pygame.QUIT:
				bullet_list = []
		keys = pygame.key.get_pressed()
		if keys[pygame.K_ESCAPE]:
			break
		turn_actions = get_turn_actions()
		if len(update_list) > max_ob:
			max_ob = len(update_list)
		for object in update_list:
			object.update()
			pass
		for list in [update_list, player_list, bullet_list, foe_list]:
			for object in [object for object in list if object.to_remove]:
				list.remove(object)
		if turn_actions >= 2:
			vframe += 1
			for object in update_list:
				object.draw()
				pass
			pygame.display.flip()
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		frame += 1
		
	l.info("Gameplay  fps : " + str(float(frame)/(pygame.time.get_ticks()-first_ticks)*1000))
	l.info("Graphical fps : " + str(float(vframe)/(pygame.time.get_ticks()-first_ticks)*1000))
	l.info( str(skip_c) + " skips")
	l.info( "max objects # : " + str(max_ob))


if __name__ == '__main__':
	main()
