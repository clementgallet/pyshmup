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

#import profile

#############
## Constants

KEY_SHOT = pygame.K_q
#KEY_SHOT = pygame.K_W

DRAW_HITBOX = True

FONKY_LINES = False

WIDTH = 640
HEIGHT = 480

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

PLAYER_SPEED = 2.0

SCALE = 400 # screen height in game units

RANK = 0.5 # difficulty setting, 0 to 1

FPS = 60
MAX_SKIP = 9

# WAIT_UNTIL_RECHECK = 30 # after checking if bullet is dangerous, wait WAIT_UNTIL_RECHECK frames

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
shot_list = []


class Foe(object):
	def __init__(self):
		self.x = UNIT_WIDTH*0
		self.y = UNIT_HEIGHT*0.3
		self.direction = 0
		self.speed = 0

		self.sprite = sprite.get_sprite( FOE_BITMAP )
		self.bullet_sprite = sprite.get_sprite( BULLET_BITMAP )
		
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
		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		   self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT):
			update_list.remove(self)
			foe_list.remove(self)
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

		new_bullet.sprite = self.bullet_sprite

	def vanish(self):
		update_list.remove(self)
		foe_list.remove(self)


class BulletMLFoe(Foe):
	#FIXME: refactor this and SimpleBulletML
	def __init__(self, bulletml_behav):
		self.aim = 0
		self.delta_x = 0
		self.delta_y = 1
		#FIXME: the first player is always aimed, ahahah !!
		if player_list:
			self.aimed_player = player_list[0]
		else:
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
			new_bullet.aimed_player = self.aimed_player
		super(BulletMLFoe, self).fire(direction, speed, new_bullet)

	def update(self):
		if self.aimed_player is not None:
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
		self.until = 0
		self.dangerous = True
		self.direction = 0
		self.speed = 0
		self.x = 0
		self.y = UNIT_HEIGHT * 0.5

		self.sprite_not_dangerous = sprite.get_sprite( BULLET_NOT_BITMAP )
		self.sprite = sprite.get_sprite (BULLET_BITMAP)
		update_list.append(self)
		bullet_list.append(self)

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

		new_bullet.sprite = self.sprite

			
	def update(self):
		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		   self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT):
			update_list.remove(self)
			bullet_list.remove(self)
		self.x += math.sin(self.direction*math.pi/180)*self.speed
		self.y -= math.cos(self.direction*math.pi/180)*self.speed
		self.t = (self.t+self.sin_spd) % (2*math.pi)


		if self.dangerous: #FIXME : Only for SimpleBullet

			if self.until > 0:
				self.until -= 1

			else:
				self.dangerous = False				

				for player in player_list:
				
					if abs(self.x - player.x) > RADIUS:
						signe_x = (self.x > player.x) + (self.x >= player.x ) - 1
						# print('signe_x = ' + str(signe_x))
						rat_x = math.sin(self.direction*math.pi/180)*self.speed - signe_x * PLAYER_SPEED
						# print('rat_x = ' + str(rat_x))
						if rat_x != 0:
							t_x = (signe_x * RADIUS - self.x + player.x) / rat_x
							# print('t_x = ' + str(t_x))
							if t_x >= 0 and - UNIT_WIDTH < player.x + signe_x * t_x * PLAYER_SPEED < UNIT_WIDTH:
								if abs(self.y - player.y) > RADIUS:
									signe_y = (self.y > player.y) + (self.y >= player.y) - 1
									# print('signe_y = ' + str(signe_y))
									rat_y = -math.cos(self.direction*math.pi/180)*self.speed - signe_y * PLAYER_SPEED
									# print('rat_y = ' + str(rat_y))
									if rat_y != 0:	
										t_y = (signe_y * RADIUS - self.y + player.y) / rat_y
										# print('t_y = ' + str(t_y))
										if t_y >= 0:
											self.dangerous = True
											self.until = int(max (t_x, t_y))

								else:
									self.dangerous = True
									self.until = int(t_x)

					else:
						if abs(self.y - player.y) > RADIUS:
							signe_y = (self.y > player.y) + (self.y >= player.y) - 1
							# print('signe_y = ' + str(signe_y))
							rat_y = -math.cos(self.direction*math.pi/180)*self.speed - signe_y * PLAYER_SPEED
							# print('rat_y = ' + str(rat_y))
							if rat_y != 0:
								t_y = (signe_y * RADIUS - self.y + player.y) / rat_y
								# print('t_y = ' + str(t_y))
								if t_y >= 0:
									self.dangerous = True
									self.until = int(t_y)

						else:
							update_list.remove(self)
							update_list.remove(player)
							bullet_list.remove(self)
							player_list.remove(player)


					#	sinv = self.speed*math.sin(self.direction*math.pi/180)
				#	cosv = self.speed*math.cos(self.direction*math.pi/180)
					
				#	A = math.sqrt(2)*PLAYER_SPEED*(player.y - self.y)
				#	B = math.sqrt(2)*PLAYER_SPEED*(player.x - self.x)
				#	C = (player.y - self.y)*sinv + (player.x - self.x)*cosv
					
					# On doit resoudre maintenant l'equation Asin(theta)+Bcos(theta) = C
					# Ce qui donne en posant X = sin(theta) :
					# (A**2 + B**2)X**2 + 2*B*C*X + C**2 - B**2 = 0 (avec -1 <= X <= 1)
					# le delta est donc : 4*B**2 (A**2 + B**2 - C**2)
					
				#	simp_delta = (A**2 + B**2 - C**2)
					
				#	if simp_delta >= 0:
						# Si le player peut rencontrer la bullet, alors
						# il faut verifier que ce ne soit pas dans les temps negatifs
						# c'est à dire que la collision ne soit pas dans le passé
						
				#		par = A * math.sqrt(simp_delta)

				#		sol1 = (B*C + par)/(B**2 + A**2) # première solution
						
				#		if (-1 <= sol1 <= 1):

				#			rat1 = (math.sqrt(2)*PLAYER_SPEED*sol1 - cosv)

				#			if rat1 == 0: # Cas bizarre
				#				self.dangerous = True
				#				dangerous_bullet_list.append(self)
				#			else:
				#				t1 = (player.y - self.y)/rat1 # temps de la collision (temps actuel = 0)
				#				if t1 > 0:
				#					if -UNIT_WIDTH*(1+OUT_LIMIT) < self.x + t1*sinv < UNIT_WIDTH*(1+OUT_LIMIT) and \
				#					       -UNIT_HEIGHT*(1+OUT_LIMIT) < self.y - t1*cosv < UNIT_HEIGHT*(1+OUT_LIMIT):
				#						self.until = WAIT_UNTIL_RECHECK #/(self.speed + 1)
				#						self.dangerous = True
				#						dangerous_bullet_list.append(self)
						
				#		if not self.dangerous:
				#			sol2 = (B*C - par)/(B**2 + A**2) # deuxieme solution
				#			if (-1 <= sol2 <= 1):
				#				rat2 = (math.sqrt(2)*PLAYER_SPEED*sol2 - cosv)
	
				#				if rat2 == 0: # Cas bizarre
				#					dangerous_bullet_list.append(self)
				#					self.dangerous = True
				#				else:
				#					t2 = (player.y - self.y)/rat2 # temps de la collision (temps actuel = 0)
				#					if t2 > 0:
				#						if -UNIT_WIDTH*(1+OUT_LIMIT) < self.x + t2*sinv < UNIT_WIDTH*(1+OUT_LIMIT) and \
				#						       -UNIT_HEIGHT*(1+OUT_LIMIT) < self.y - t2*cosv < UNIT_HEIGHT*(1+OUT_LIMIT):
				#							self.until = WAIT_UNTIL_RECHECK #/(self.speed + 1)
				#							self.dangerous = True
				#							dangerous_bullet_list.append(self)

				

					
	def draw(self):
		glPushMatrix()
		glColor4f(1.0, 1.0, 1.0, 0.2)
		glTranslatef(self.x, self.y, 0)#math.sin(self.t)*5)
		# glRotatef(self.t * 180/math.pi, 0, 0, 1)
		if self.dangerous:
			self.sprite.draw()
		else:
			self.sprite_not_dangerous.draw()
		glPopMatrix()
		#glTranslatef(-self.x, -self.y, 0)

	def vanish(self):
		update_list.remove(self)
		bullet_list.remove(self)


class SimpleBulletML(SimpleBullet):
	def __init__(self, bulletml_behav=None):
		self.aim = 0
		self.delta_x = 0
		self.delta_y = 0
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
			new_bullet.aimed_player = self.aimed_player
		super(SimpleBulletML, self).fire(direction, speed, new_bullet)

	def update(self):
		if self.aimed_player is not None:
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

		
		

SHOT_IN_SPEED = 4
SHOT_OUT_SPEED = 10
SHOT_IN_WIDTH = 5
SHOT_OUT_WIDTH = 20
SHOT_SPIN = 20

class Shot:

	x = 0
	y = 0
	frame = 0
	initial_x = 0
	width = 10
	speed = SHOT_IN_SPEED
	def __init__(self):
		self.sprite = sprite.get_sprite( BULLET_BITMAP )
		shot_list.append(self)
		update_list.append(self)
		
	def update(self):
		if self.x < -UNIT_WIDTH*(1+OUT_LIMIT)  or self.x > UNIT_WIDTH*(1+OUT_LIMIT) or \
		   self.y < -UNIT_HEIGHT*(1+OUT_LIMIT) or self.y > UNIT_HEIGHT*(1+OUT_LIMIT):
			update_list.remove(self)
			shot_bullet.remove(self)
		self.y += self.speed
		self.x = math.cos(float(self.frame)*math.pi/(2*SHOT_SPIN))*self.width + self.initial_x

	def draw(self):
		glPushMatrix()
		glColor4f(1.0, 1.0, 1.0, 0.2)
		glTranslatef(self.x, self.y, 0)#math.sin(self.t)*5)
		# glRotatef(self.t * 180/math.pi, 0, 0, 1)
		self.sprite.draw()
		glPopMatrix()
		#glTranslatef(-self.x, -self.y, 0)

	def vanish(self):
		update_list.remove(self)
		shot_list.remove(self)


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
		#s = ([[b.x,b.y,self.x, self.y, (b.x-self.x)**2+(b.x-self.y)**2] for b in bullet_list])# < RADIUS:
		#random.shuffle(s)
		#print s[0]
		#if dangerous_bullet_list:
		#	if min([(b.x-self.x)**2+(b.y-self.y)**2 for b in dangerous_bullet_list]) < RADIUS2:
		#		self.to_remove = True


		if keys[KEY_SHOT]:
			shot = Shot()
			shot.initial_x = self.x
			shot.y = self.y
			shot.frame = self.frame
			shot.width = SHOT_IN_WIDTH
			shot.speed = SHOT_IN_SPEED
			shot.x = math.cos(float(shot.frame)*math.pi/SHOT_SPIN)*shot.width + shot.initial_x
				
			shot = Shot()
			shot.initial_x = self.x
			shot.y = self.y
			shot.frame = self.frame + 2*SHOT_SPIN
			shot.speed = SHOT_IN_SPEED
			shot.width = SHOT_IN_WIDTH
			shot.x = math.cos(float(shot.frame)*math.pi/SHOT_SPIN)*shot.width + shot.initial_x
			
			shot = Shot()
			shot.initial_x = self.x
			shot.y = self.y
			shot.frame = self.frame + SHOT_SPIN
			shot.width = SHOT_OUT_WIDTH
			shot.speed = SHOT_OUT_SPEED
			shot.x = math.cos(float(shot.frame)*math.pi/SHOT_SPIN)*shot.width + shot.initial_x

			shot = Shot()
			shot.initial_x = self.x
			shot.y = self.y
			shot.frame = self.frame + 3*SHOT_SPIN
			shot.width = SHOT_OUT_WIDTH
			shot.speed = SHOT_OUT_SPEED
			shot.x = math.cos(float(shot.frame)*math.pi/SHOT_SPIN)*shot.width + shot.initial_x

				
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
		

class Stage:
	def __init__(self):
		self.foe_list = copy.deepcopy(StagetoFoeList('stage.xml').getFoes())
		self.frame = 0

	def update(self):
		# print(str(self.foe_list[0].frame) + ' | ' + str(self.frame))
		while (self.foe_list and self.foe_list[0].frame == self.frame):
			# print('launch')
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
#	BulletMLFoe(FILE)
	stage = Stage()
#	while bullet_list or foe_list:
	while pygame.time.get_ticks() < 50000:

		# print(str(len(dangerous_bullet_list)) + " - " + str(len(bullet_list)))
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
		stage.update()
		for object in update_list:
			object.update()
			pass

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
