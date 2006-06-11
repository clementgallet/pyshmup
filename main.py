# -*- coding: utf-8 -*-

import bulletml
from bulletml import BulletMLController
import sprite
import stage
from stage import StagetoFoeList

NO_GRAPHICS = False

import pygame
import logging
import math
import time
import random
import copy
from OpenGL.GL import * # evil
from OpenGL.GLU import * 
from Numeric import *
import draw

#import profile

#############
## Constants

KEY_SHOT = pygame.K_q
#KEY_SHOT = pygame.K_W

DRAW_HITBOX = True

FONKY_LINES = False

NO_DEATH = True

WIDTH = 640
HEIGHT = 480

STAGE_FILE = "stage.xml"
#STAGE_FILE = "stage2.xml"
BITMAP_PATH = "data/images/"
BEHAV_PATH = "data/bullets/"
SHIP_BITMAP = "data/images/ship.png"
#SHIP_BITMAP = "data/images/shot_small.png"
BULLET_BITMAP = "data/images/shot.png"
BULLET_NOT_BITMAP = "data/images/shot2.png"
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
FOE_RADIUS = 10.0 # foe's "hit-disc" radius, in game units

PLAYER_SPEED = 2.0

FOE_LIFE = 30

NB_LINES = 20 # number of lines for the shot

SCALE = 400 # screen height in game units

bulletml.RANK = 0.5 # difficulty setting, 0 to 1

FPS = 60
MAX_SKIP = 9

ARRAY_DIM = 6

SHOT_TIME = 40
SHOT_FAST = 1 # in presses every SHOT_TIME frames

BACKGROUND_COLOR = (.235, .275, .275, 1)

SINUS_LIST = [math.sin(i*math.pi / 1800) for i in range(3601)]
COSINUS_LIST = [math.cos(i*math.pi / 1800) for i in range(3601)]

#####################
## Derived constants

RADIUS2 = RADIUS * RADIUS

UNIT_HEIGHT = SCALE/2
UNIT_WIDTH = (UNIT_HEIGHT * WIDTH) / HEIGHT

if FONKY_LINES:
	LINE_RADIUS2 = 10000*RADIUS2
	FONKY_COLOR = [.205, .245, .245, 1]
	FONKY_OFFSET = [ BACKGROUND_COLOR[i] - FONKY_COLOR[i] for i in [0,1,2] ]

SHOT_COLOR = [.425, .475, .475, 1]

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
	glBlendFunc(GL_SRC_ALPHA, GL_ONE)#MINUS_SRC_ALPHA)
	glEnable(GL_BLEND)

	glEnable(GL_TEXTURE_2D)

	pygame.display.set_caption("FIVE TONS OF FLAX !")
	


# this is not intended to be particularly exten{ded,sible}

update_list = []
bullet_list = []
bullet_noml_list = []
player_list = []
foe_list = []
shot_list = []

bullet_array = zeros((ARRAY_DIM,8),Float)
ARRAY_X = 0
ARRAY_Y = 1
ARRAY_Z = 2
ARRAY_DIRECTION = 3
ARRAY_SPEED = 4
ARRAY_CALLLIST = 5
ARRAY_DIM = 6

array_fill = 0
array_size = 8

class Foe(object):
	def __init__(self):
		self.x = UNIT_WIDTH*0
		self.y = UNIT_HEIGHT*0.3
		self.z = 0
		self.direction = 0
		self.speed = 0
		self.to_remove = False
		self.sprite = sprite.get_sprite( FOE_BITMAP )
		self.bullet_sprite = sprite.get_sprite( BULLET_BITMAP )
		self.life = FOE_LIFE
		update_list.append(self)
		foe_list.append(self)
		
	def draw(self):
		glPushMatrix()
		# TODO: if foe has no target, it becomes transparent
		glColor4f(1.0, 1.0, 1.0, 1.0)
		glTranslatef(self.x, self.y, 0)
		self.sprite.draw()
		glPopMatrix()


	def update(self):
		self.direction = self.direction % 360
		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		   self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT) or self.life < 0:
			foe_list.remove(self)
			self.to_remove = True
		self.x += SINUS_LIST[int(10*self.direction)]*self.speed
		self.y -= COSINUS_LIST[int(10*self.direction)]*self.speed

		for shot in shot_list:
			if max(abs(shot.x - self.x),abs(shot.y - self.y)) < FOE_RADIUS:
				self.life -= 1
				shot.vanish()
		
		return self

	def fireml(self, direction=None, speed=None, new_bullet=None):
		if direction is not None:
			new_bullet.direction = direction
		else:
			new_bullet.direction = self.direction
		if speed is not None:
			new_bullet.speed = speed
		else:
			new_bullet.speed = self.speed
			
		bullet_array[:5,new_bullet.index] = [self.x,self.y,self.z,new_bullet.direction,new_bullet.speed]
		new_bullet.sprite = self.bullet_sprite


	def firenoml(self, direction=None, speed=None, new_bullet=None):
		self.z -= 0.0001
		if direction is not None:
			new_bullet.direction = direction
		else:
			new_bullet.direction = self.direction
		if speed is not None:
			new_bullet.speed = speed
		else:
			new_bullet.speed = self.speed
			
		if abs(new_bullet.direction % 180) < 0.1 or new_bullet.speed == 0:
			time_x = 10000000
		elif 0 < new_bullet.direction % 360 < 180:
			time_x = (UNIT_WIDTH*(1+OUT_LIMIT) - self.x)/(SINUS_LIST[int(10*(new_bullet.direction % 360))]*new_bullet.speed)
		else:
			time_x = (-UNIT_WIDTH*(1+OUT_LIMIT) - self.x)/(SINUS_LIST[int(10*(new_bullet.direction % 360))]*new_bullet.speed)

		if abs((new_bullet.direction % 180) - 90) < 0.1 or new_bullet.speed == 0:
			time_y = 10000000
		elif 90 < new_bullet.direction % 360 < 270:
			time_y = (UNIT_HEIGHT*(1+OUT_LIMIT) - self.y)/(-COSINUS_LIST[int(10*(new_bullet.direction % 360))]*new_bullet.speed)
		else:
			time_y = (-UNIT_HEIGHT*(1+OUT_LIMIT) - self.y)/(-COSINUS_LIST[int(10*(new_bullet.direction % 360))]*new_bullet.speed)

		new_bullet.out_time = int(min(time_x,time_y))

		bullet_array[:ARRAY_DIRECTION,new_bullet.index] = [self.x,self.y,self.z+0.00001,new_bullet.direction,new_bullet.speed]
		new_bullet.sprite = self.bullet_sprite



	def vanish(self):
		foe_list.remove(self)
		self.to_remove = True

class BulletMLFoe(Foe):
	#FIXME: refactor this and SimpleBulletML
	def __init__(self, bulletml_behav):

		#FIXME: the first player is always aimed, ahahah !!
		if player_list:
			self.aimed_player = player_list[0]
		else:
			self.aimed_player = None
		self.controller = BulletMLController()
		self.controller.game_object = self
		self.controller.set_behavior(bulletml_behav)
		super(BulletMLFoe, self).__init__()

	def fireml(self, controller, direction=None, speed=None):
		new_bullet = SimpleBulletML()
		new_bullet.controller = controller
		controller.set_game_object(new_bullet)
		new_bullet.aimed_player = self.aimed_player
		super(BulletMLFoe, self).fireml(direction, speed, new_bullet)


	def firenoml(self, direction=None, speed=None):
		new_bullet = SimpleBulletNoML()
		super(BulletMLFoe, self).firenoml(direction, speed, new_bullet)


	def update(self):
		self.controller.run()

		return super(BulletMLFoe, self).update()

class SimpleBullet(object):
	def __init__(self):
		global bullet_array,array_size,array_fill,to_remove_array
		self.direction = 0
		self.speed = 1
		self.index = array_fill
		array_fill += 1
		if array_fill == array_size:
			new_array = zeros((ARRAY_DIM,2*array_size),Float)
			new_array[:,:array_size] = bullet_array
			bullet_array = new_array
			array_size *= 2
		
		self.to_remove = False
		bullet_list.append(self)
		self.sprite = sprite.get_sprite (BULLET_BITMAP)
		bullet_array[5][self.index] = self.sprite.list
		#self.t = 0
		#self.sin_spd = random.random() * 0.04
	
	def getx(self):
		return bullet_array[0,self.index]
	x = property(getx)
	def gety(self):
		return bullet_array[1,self.index]
	y = property(gety)
	def getz(self):
		return bullet_array[2,self.index]
	z = property(getz)
		
					
	def draw(self):
		#self.t = (self.t+self.sin_spd) % (2*math.pi)
		glPushMatrix()
		glColor4f(1.0, 1.0, 1.0, 0.2)

	      	self.x,self.y = bullet_array[:2,self.index]

		glTranslatef(self.x, self.y, 0)#math.sin(self.t)*5)
		# glRotatef(self.t * 180/math.pi, 0, 0, 1)
		self.sprite.draw()
		glPopMatrix()
		#glTranslatef(-self.x, -self.y, 0)

	def vanish(self):
		global array_fill
		
		self.to_remove = True
		array_fill -= 1
		if array_fill == self.index:
			bullet_list.pop()
		else:
			bullet_array[:,self.index] = bullet_array[:,array_fill]
			bullet_list[self.index] = bullet_list.pop()
			bullet_list[self.index].index = self.index

		
class SimpleBulletNoML(SimpleBullet):

	def __init__(self):
		self.until = 0
		self.dangerous = True
		self.out_time = 0
		bullet_noml_list.append(self)
		super(SimpleBulletNoML, self).__init__()


	def vanish(self):
		bullet_noml_list.remove(self)
		super(SimpleBulletNoML, self).vanish()



class SimpleBulletML(SimpleBullet):

	def __init__(self, bulletml_behav=None):

		update_list.append(self)
		self.controller = BulletMLController()
		self.controller.game_object = self
		if bulletml_behav is not None:
			self.controller.set_behavior(bulletml_behav)
		super(SimpleBulletML, self).__init__()

	def fireml(self, controller, direction=None, speed=None):
		new_bullet = SimpleBulletML()
		new_bullet.controller = controller
		controller.set_game_object(new_bullet)
		new_bullet.aimed_player = self.aimed_player


		if direction is not None:
			new_bullet.direction = direction
		else:
			new_bullet.direction = self.direction
		if speed is not None:
			new_bullet.speed = speed
		else:
			new_bullet.speed = self.speed
			
		bullet_array[:5,new_bullet.index] = self.x,self.y,self.z+0.0001,new_bullet.direction,new_bullet.speed
		new_bullet.sprite = self.sprite

	def firenoml(self, direction=None, speed=None):
		new_bullet = SimpleBulletNoML()
		new_bullet.aimed_player = self.aimed_player

		if direction is not None:
			new_bullet.direction = direction
		else:
			new_bullet.direction = self.direction
		if speed is not None:
			new_bullet.speed = speed
		else:
			new_bullet.speed = self.speed
			
		if abs(new_bullet.direction % 180) < 0.1 or new_bullet.speed == 0:
	       		time_x = 10000000
	       	elif 0 < new_bullet.direction % 360 < 180:
	       		time_x = (UNIT_WIDTH*(1+OUT_LIMIT) - self.x)/(SINUS_LIST[int(10*(new_bullet.direction % 360))]*new_bullet.speed)
	       	else:
       			time_x = (-UNIT_WIDTH*(1+OUT_LIMIT) - self.x)/(SINUS_LIST[int(10*(new_bullet.direction % 360))]*new_bullet.speed)

       		if abs((new_bullet.direction % 180) - 90) < 0.1 or new_bullet.speed == 0:
       			time_y = 10000000
       		elif 90 < new_bullet.direction % 360 < 270:
       			time_y = (UNIT_HEIGHT*(1+OUT_LIMIT) - self.y)/(-COSINUS_LIST[int(10*(new_bullet.direction % 360))]*new_bullet.speed)
       		else:
       			time_y = (-UNIT_HEIGHT*(1+OUT_LIMIT) - self.y)/(-COSINUS_LIST[int(10*(new_bullet.direction % 360))]*new_bullet.speed)

       		new_bullet.out_time = int(min(time_x,time_y))
		bullet_array[:5,new_bullet.index] = self.x,self.y,self.z,new_bullet.direction,new_bullet.speed
		new_bullet.sprite = self.sprite

	def update(self):
		
		self.controller.run()

		if self.to_remove:
			return self
		else:
			bullet_array[3:5,self.index] = self.direction % 360, self.speed

		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		       self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT):
			self.vanish()
			return self
			
		for player in player_list:
			if max(abs(self.x - player.x),abs(self.y - player.y)) <= RADIUS:
				player.vanish()
				self.vanish()

		return self	


SHOT_WIDTH = 50

class Shot:

	def __init__(self):

		self.x = 0
		self.y = 0
		self.lines = []

		shot_list.append(self)
		update_list.append(self)
		
		self.to_remove = False
		self.aimed_foe = None
		

		
	def update(self):

		if not self.aimed_foe in foe_list and not self.to_remove:
			self.vanish()
			return self

		dist = (self.x - self.aimed_foe.x)**2 + (self.y - self.aimed_foe.y)**2

		xpos = ((self.aimed_foe.x > self.x) + (self.aimed_foe.x >= self.x) - 1)*(((self.aimed_foe.x - self.x)**2)/dist) + 1
		xneg = 2 - xpos
		ypos = ((self.aimed_foe.y > self.y) + (self.aimed_foe.y >= self.y) - 1)*(((self.aimed_foe.y - self.y)**2)/dist) + 1
		yneg = 2 - ypos

		xpos *= xpos
		xneg *= xneg
		ypos *= ypos
		yneg *= yneg
		
		#print (str(xpos) + ' - ' + str(xneg) + ' - ' + str(ypos) + ' - ' + str(yneg))

		choix = random.random()*(xpos + xneg + ypos + yneg)

		shot_dist = math.sqrt(dist)/2
		
		if choix < xpos:
			self.x += shot_dist
		elif xpos <= choix < xneg + xpos:
			self.x -= shot_dist
		elif xneg + xpos <= choix < xneg + xpos + ypos:
			self.y += shot_dist
		else:
			self.y -= shot_dist

		self.lines.append((self.x,self.y))

		return self
	
	def draw(self):
		#glPushMatrix()

		if len(self.lines) > NB_LINES:
			self.lines.pop(0)
			
		taille = len(self.lines) - 1
		c = 0

		if taille >= 0:
			glDisable(GL_TEXTURE_2D)
			for i in self.lines:
				(x,y) = i
				try:
					pen_x = pen_x
					SHOT_COLOR[3] = 1 - (float(taille - c) / NB_LINES)
					glBegin(GL_LINES)
					glColor4f(*SHOT_COLOR)
					glVertex2f(x, y)
					glVertex2f(pen_x, pen_y)
					glEnd()
				except:
					pass
				
				pen_x = x
				pen_y = y
				
				c += 1
			glColor4f(1.0, 1.0, 1.0, 1.0)


	def vanish(self):
		shot_list.remove(self)
		self.to_remove = True


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
		self.frame += 1

		for bullet in bullet_noml_list:

			if bullet.out_time == 0:
				bullet.vanish()
			bullet.out_time -= 1
			
			if bullet.dangerous:

				if bullet.until > 0:
					bullet.until -= 1

				else:
					bullet.dangerous = False				
					x,y = bullet_array[:2,bullet.index]
										
					if abs(x - self.x) > RADIUS:
						signe_x = (x > self.x) + (x >= self.x ) - 1
						rat_x = SINUS_LIST[int(10*(bullet.direction % 360))]*bullet.speed - signe_x * PLAYER_SPEED
						if rat_x != 0:
							t_x = (signe_x * RADIUS - x + self.x) / rat_x
							if t_x >= 0 and - UNIT_WIDTH < self.x + signe_x * t_x * PLAYER_SPEED < UNIT_WIDTH:
								if abs(y - self.y) > RADIUS:
									signe_y = (y > self.y) + (y >= self.y) - 1
									rat_y = -COSINUS_LIST[int(10*(bullet.direction % 360))]*bullet.speed - signe_y * PLAYER_SPEED
									if rat_y != 0:	
										t_y = (signe_y * RADIUS - y + self.y) / rat_y
										if t_y >= 0:
											bullet.dangerous = True
											bullet.until = int(max (t_x, t_y))

								else:
									bullet.dangerous = True
									bullet.until = int(t_x)
					else:
						if abs(y - self.y) > RADIUS:
							signe_y = (y > self.y) + (y >= self.y) - 1
							rat_y = -COSINUS_LIST[int(10*(bullet.direction % 360))]*bullet.speed - signe_y * PLAYER_SPEED
							if rat_y != 0:
								t_y = (signe_y * RADIUS - y + self.y) / rat_y
								if t_y >= 0:
									bullet.dangerous = True
									bullet.until = int(t_y)

						else:
							self.vanish()
							bullet.vanish()


		if keys[KEY_SHOT]:
			
			foe_aimed_list = []
			for foe in foe_list:
				if foe.y > self.y and abs(foe.x - self.x) < SHOT_WIDTH / 2:
					foe_aimed_list.append(foe)

			if foe_aimed_list:
				foe_choosen = random.randint(0,len(foe_aimed_list) - 1)
				shot = Shot()
				shot.x = self.x
				shot.y = self.y
				shot.aimed_foe = foe_aimed_list[foe_choosen]


		return self
		# "front montant"
		#shot_pressed = (not self.last_shot_pressed) and keys[KEY_SHOT]
		#self.last_shot_pressed = keys[KEY_SHOT]
		
		# find shot_state
		# if self.shot_state == 0:
		#	if shot_pressed:
		#		self.to_next_shot_limit = SHOT_TIME
		#		self.shot_count = 0
		#		self.last_shot_state = 0
		#		self.shot_state = 1
		#else:
		#	if self.to_next_shot_limit <= 0:
		#		# we have to decide the next firepower
		#		if self.shot_count >= self.shot_state:
		#			self.last_shot_state = self.shot_state
		#			if self.shot_state < SHOT_HIGH:
		#				self.shot_state += 1
		#		elif self.shot_count > 0:
		#			# statu quo
		#			self.last_shot_state = self.shot_state
		#		else:
		#			if self.last_shot_state >= self.shot_state:
		#				self.last_shot_state = self.shot_state
		#				if self.shot_state > 1:
		#					self.shot_state -= 1
		#				else:
		#					if self.shot_count > 0:
		#						self.last_shot_state = 1 # staying in LOW anyway
		#					else:
		#						self.shot_state = SHOT_NO
		#			else:
		#				self.last_shot_state = self.shot_state
		#		self.to_next_shot_limit = SHOT_TIME
		#		self.shot_count = 0
		#		if shot_pressed:
		#			self.shot_count = 1
		#	else:
		#		if shot_pressed:
		#			self.shot_count += 1
		#		self.to_next_shot_limit -= 1

		#print "state ", self.shot_state, " for ", self.to_next_shot_limit, " (", self.shot_count, ") [", \
		#	shot_pressed, "]"
		#global BACKGROUND_COLOR
#		#BACKGROUND_COLOR = ( 50* self.shot_state, 50* self.shot_state, 50* self.shot_state )

		#TODO: show it !
								

	def __init__(self):
		self.x = 0.0
		self.y = -UNIT_HEIGHT * .5
		self.frame = 0
		update_list.append(self)
		player_list.append(self)
		self.to_remove = False
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
			glDisable(GL_TEXTURE_2D)
			for i in range(array_fill):
				x,y = bullet_array[:2,i]
				coeff = ((float(self.x)-x)**2+(self.y-y)**2)/LINE_RADIUS2
				if coeff <= 1:
					FONKY_COLOR[3] = (1-coeff) ** 2 # alpha component
					glColor4f(*FONKY_COLOR)
					glBegin(GL_LINES)
					glVertex2f(x, y)
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
		


	def vanish(self):
		if not NO_DEATH:
			self.to_remove = True
			player_list.remove(self)
		else:
			pass

class Stage:
	def __init__(self):
		self.foe_list = copy.deepcopy(StagetoFoeList(STAGE_FILE).getFoes())
		self.frame = 0

	def update(self):
		while (self.foe_list and self.foe_list[0].frame == self.frame):
			foe = self.foe_list[0]
			self.launch(foe.behav,foe.x,foe.y,foe.sprite,foe.bullet)
			self.foe_list.remove(foe)
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
	global update_list,bullet_list
	init_sdl()
	x=0
	first_ticks = pygame.time.get_ticks()
	frame = 0
	vframe = 0
	max_ob = 0
	Player()
#	SimpleBulletML(FILE)
#	Foe()
#	BulletMLFoe(FILE)
	stage = Stage()
	first_fps = first_ticks
	frame_loc = 0
#	while bullet_list or foe_list:
	while pygame.time.get_ticks() < 50000:

		frame_loc += 1

		if pygame.time.get_ticks() - first_fps >= 1000:
			pygame.display.set_caption("FIVE TONS OF FLAX ! - fps : " + str(frame_loc))
			first_fps = pygame.time.get_ticks()
			frame_loc = 0

		for ev in pygame.event.get():
			if ev.type == pygame.VIDEORESIZE:
				set_perspective(ev.w, ev.h)
			if ev.type == pygame.QUIT:
				bullet_list = []
		keys = pygame.key.get_pressed()
		if keys[pygame.K_ESCAPE]:
			break
		turn_actions = get_turn_actions()
		if len(update_list) + len(bullet_noml_list) > max_ob:
			max_ob = len(update_list) + len(bullet_noml_list)
		stage.update()

		update_list = [obj for obj in update_list if obj.update().to_remove == False]

		add(bullet_array[0],multiply(sin(multiply(bullet_array[3],math.pi/180)),bullet_array[4]),bullet_array[0])
		subtract(bullet_array[1],multiply(cos(multiply(bullet_array[3],math.pi/180)),bullet_array[4]),bullet_array[1])

		if turn_actions >= 2:
			vframe += 1
			for object in player_list:
				object.draw()
			for object in foe_list:
				object.draw()
			for object in shot_list:
				object.draw()
			draw.draw(bullet_array,array_fill)
			pygame.display.flip()
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		frame += 1
		
	l.info("Gameplay  fps : " + str(float(frame)/(pygame.time.get_ticks()-first_ticks)*1000))
	l.info("Graphical fps : " + str(float(vframe)/(pygame.time.get_ticks()-first_ticks)*1000))
	l.info( str(skip_c) + " skips")
	l.info( "max objects # : " + str(max_ob))

#profile.run('main()','prof.txt')

if __name__ == '__main__':
	main()
