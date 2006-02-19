# -*- coding: utf-8 -*-

import bulletparse
from bulletparse import GameObjectMainController

NO_GRAPHICS = False

import pygame
import logging
import math
import time
import random
import copy

import profile

#############
## Constants

KEY_SHOT = pygame.K_q
#KEY_SHOT = pygame.K_W

PARTIAL_UPDATE = True

DRAW_HITBOX = True

FONKY_LINES = True

WIDTH = 200
HEIGHT = 150
#SHIP_BITMAP = "data/images/ship.png"
SHIP_BITMAP = "data/images/shot_small.png"
#ENEMY_SHOT_BITMAP = "data/images/shot.png"
ENEMY_SHOT_BITMAP = "data/images/shot_small.png"
#FILE = "data/bullets/doud.xml"
#FILE = "data/bullets/doud_synch.xml"
#FILE = "data/bullets/doud_directed_rand_smooth.xml"
#FILE = "data/bullets/bullets/doud_circles.xml"
FILE = 'data/bullets/struggle.xml'
#FILE = 'data/bullets/bee.xml'
#FILE = 'data/bullets/slowdown.xml'
#FILE = 'data/bullets/beewinder.xml'
#FILE = 'data/bullets/side_cracker.xml'
#FILE = 'data/data/bullets/roll3pos.xml'
#FILE = 'data/bullets/rollbar.xml'
#FILE = 'data/bullets/bullets/keylie_360.xml'
#FILE = 'data/bullets/double_roll_seeds.xml'
#FILE = 'data/bullets/[ketsui]_r4_boss_rb_rockets.xml'
#FILE = 'data/bullets/quad3.xml'
#FILE = 'data/bullets/roll.xml'
#FILE = 'data/bullets/4waccel.xml'
#FILE = 'data/bullets/248shot.xml'
#FILE = 'data/data/bullets/bar.xml'
#FILE = 'data/bullets/[Ikaruga]_r5_vrp.xml'
FILE = 'data/bullets/[Progear]_round_3_boss_back_burst.xml'

OUT_LIMIT = 0.2

RADIUS = 3

SCALE = 1.0/500

RANK = 1.0

FPS = 60
MAX_SKIP = 9

SHOT_TIME = 40
SHOT_FAST = 1 # in presses every SHOT_TIME frames

BACKGROUND_COLOR = (60, 70, 70)

#####################
## Derived constants

SIZE_FACTOR = SCALE * WIDTH

RADIUS2_PIXELS = (RADIUS*SIZE_FACTOR)**2
LINE_RADIUS2_PIXELS = 1600*RADIUS2_PIXELS

if FONKY_LINES:
	PARTIAL_UPDATE = False
	FONKY_COLOR = [50, 60, 65]
	FONKY_OFFSET = [ BACKGROUND_COLOR[i] - FONKY_COLOR[i] for i in [0,1,2] ]

if PARTIAL_UPDATE:
	BACKGROUND_COLOR = (0, 0, 0)

############
## Logging

console = logging.StreamHandler( )
console.setLevel( logging.DEBUG )
formatter = logging.Formatter( '%(name)-12s: %(levelname)-8s %(message)s' )
console.setFormatter( formatter )
logging.getLogger('').addHandler( console )

l = logging.getLogger('main')
l.setLevel(logging.DEBUG)

images = {}

bulletparse.RANK = RANK

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
	screen = pygame.display.set_mode( (WIDTH,HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE )
	pygame.display.set_caption("TOUCHED BY HIS NOODLY APPENDAGE")
	
		
																						
# this is not intended to be particurly exten{ded,sible}
update_list = []
bullet_list = []
player_list = []

class SimpleBullet:
	direction = 0.0
	speed = 0.0
	x = WIDTH / 2.0
	y = HEIGHT / 2.0

	def __init__(self):
		self.controller = GameObjectMainController()
		self.controller.game_object = self
		self.controller.set_behavior(FILE)

		self.image = get_image( ENEMY_SHOT_BITMAP )
		self.rect = self.image.get_rect()
		self.rect.center = (self.x, self.y)
		if PARTIAL_UPDATE:
			self.last_rect = copy.deepcopy(self.rect)
			self.temp_rect = copy.deepcopy(self.rect)

		update_list.append(self)
		bullet_list.append(self)
		self.to_remove = False


	def fire(self, controller, direction=None, speed=None):
		new_bullet = SimpleBullet()
		new_bullet.controller = controller 
		controller.set_game_object(new_bullet)
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
		if self.x < -WIDTH*OUT_LIMIT  or self.x > WIDTH*(1+OUT_LIMIT) or \
		   self.y < -HEIGHT*OUT_LIMIT or self.y > HEIGHT*(1+OUT_LIMIT):
			self.to_remove = True
		self.controller.run()
		self.x += math.sin(self.direction*math.pi/180)*self.speed*SIZE_FACTOR
		self.y += math.cos(self.direction*math.pi/180)*self.speed*SIZE_FACTOR
		self.rect.center = (self.x, self.y)

	def draw(self):
		screen.blit(self.image, self.rect)
		if PARTIAL_UPDATE:
			self.last_rect.center = self.temp_rect.center
			self.temp_rect.center = self.rect.center

	def vanish(self):
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
			dy -= 1
		if keys[pygame.K_DOWN]:
			dy += 1
		self.x += dx*WIDTH*SCALE
		self.y += dy*WIDTH*SCALE
		self.rect.center=(self.x,self.y)
		#s = ([[b.x,b.y,self.x, self.y, (b.x-self.x)**2+(b.x-self.y)**2] for b in bullet_list])# < RADIUS:
		#random.shuffle(s)
		#print s[0]
		if min([(b.x-self.x)**2+(b.y-self.y)**2 for b in bullet_list]) < RADIUS2_PIXELS:
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
		self.image = get_image(SHIP_BITMAP)
		self.rect = self.image.get_rect()
		self.temp_rect = copy.deepcopy(self.rect)
		self.last_rect = copy.deepcopy(self.rect)
		self.x = WIDTH/2
		self.y = HEIGHT*4/5
		self.rect.center = (self.x, self.y)
		self.to_remove = False
		update_list.append(self)
		player_list.append(self)

		self.shot_state = SHOT_NO
		self.to_next_shot_limit = 0
		self.last_shot_pressed = False
		self.shot_pressed = False
		self.shot_count = 0
		
	def draw(self):
		if FONKY_LINES:
			for b in bullet_list:
				coeff = ((float(self.x)-b.x)**2+(self.y-b.y)**2)/LINE_RADIUS2_PIXELS
				if coeff <= 1:
					color = [ FONKY_COLOR[i] + FONKY_OFFSET[i] * coeff for i in [0,1,2] ]
					pygame.draw.line(screen, color, (self.x, self.y), (b.x, b.y))
		if DRAW_HITBOX and FONKY_LINES:
			pygame.draw.circle(screen, FONKY_COLOR, (int(self.x), int(self.y)), int(RADIUS*SIZE_FACTOR))
		screen.blit(self.image, self.rect)
		self.last_rect.center = self.temp_rect.center
		self.temp_rect.center = self.rect.center

fskip = 0
def get_turn_actions():
	global old_ticks
	global skip_c, fskip
	new_ticks = pygame.time.get_ticks()
	next_ticks = old_ticks + 1000/FPS
	old_ticks = next_ticks
	if new_ticks <= next_ticks or fskip > MAX_SKIP:
		pygame.time.wait(next_ticks - new_ticks)
		fskip = 0
		return 2
	else:
		skip_c += 1
		fskip += 1
	if fskip > MAX_SKIP:
		next_ticks = new_ticks + 1000/FPS

skip_c = 0
def main():
	global old_ticks
	init_sdl()
	x=0
	first_ticks = pygame.time.get_ticks()
	frame = 0
	vframe = 0
	max_ob = 0
	old_ticks = pygame.time.get_ticks()
	Player()
	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
	last_rects = []
	while bullet_list:
		while pygame.event.get():
			pass
		turn_actions = get_turn_actions()
		if len(update_list) > max_ob:
			max_ob = len(update_list)
		for object in update_list:
			object.update()
		for list in [update_list, player_list, bullet_list]:
			for object in [object for object in list if object.to_remove]:
				list.remove(object)
				if PARTIAL_UPDATE and list == update_list:
					pygame.display.update(object.temp_rect)
		if turn_actions >= 2:
			vframe += 1
			for object in update_list:
				object.draw()
			if PARTIAL_UPDATE:
				pygame.display.update([b.rect for b in update_list] + [b.last_rect for b in update_list])
			else:
				pygame.display.flip()
			screen.fill( BACKGROUND_COLOR )
		frame += 1
		
		#pygame.display.update(pygame.Rect(200,200,100,100))
		#time.sleep(.02)
		#print bullet.x, bullet.y
	print "Gameplay  fps :", float(frame)/(pygame.time.get_ticks()-first_ticks)*1000
	print "Graphical fps :", float(vframe)/(pygame.time.get_ticks()-first_ticks)*1000
	print skip_c, "skips"
	print "max objects # :", max_ob


if __name__ == '__main__':
	main()
