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

#############
## Constants

PARTIAL_UPDATE = True

WIDTH = 640
HEIGHT = 480
#SHOT_BITMAP = "data/images/shot.png"
SHOT_BITMAP = "data/images/shot_small.png"
#FILE = "doud.xml"
#FILE = "doud_synch.xml"
#FILE = "doud_directed_rand_smooth.xml"
#FILE = 'struggle.xml'
FILE = 'bee.xml'
#FILE = 'slowdown.xml'
#FILE = 'beewinder.xml'
#FILE = 'side_cracker.xml'
#FILE = 'roll3pos.xml'
#FILE = 'rollbar.xml'
#FILE = 'keylie_360.xml'
OUT_LIMIT = 0.2

SPEED_FACTOR = 1.0/400

RANK = 1

FPS = 60
MAX_SKIP = 9

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
	pygame.display.set_caption("out of funny title ideas...")
	
		
																						

update_list = []

class SimpleBullet:
	direction = 0.0
	speed = 0.0
	x = WIDTH / 2.0
	y = HEIGHT / 2.0

	def __init__(self):
		self.controller = GameObjectMainController()
		self.controller.game_object = self
		self.controller.set_behavior(FILE)

		self.image = get_image( SHOT_BITMAP )
		self.rect = self.image.get_rect()
		self.rect.center = (self.x, self.y)
		if PARTIAL_UPDATE:
			self.last_rect = copy.deepcopy(self.rect)
			self.temp_rect = copy.deepcopy(self.rect)

		update_list.append(self)
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
		if self.x < -WIDTH*OUT_LIMIT or self.x > WIDTH*(1+OUT_LIMIT) or self.y < -HEIGHT*OUT_LIMIT or self.y > HEIGHT*(1+OUT_LIMIT):
			self.to_remove = True
		self.controller.run()
		self.x += math.cos(self.direction*math.pi/180)*self.speed*WIDTH*SPEED_FACTOR*3/4
		self.y -= math.sin(self.direction*math.pi/180)*self.speed*HEIGHT*SPEED_FACTOR
		self.rect.center = (self.x, self.y)

	def draw(self):
		screen.blit(self.image, self.rect)
		if PARTIAL_UPDATE:
			self.last_rect.center = self.temp_rect.center
			self.temp_rect.center = self.rect.center

	def vanish(self):
		self.to_remove = True
		
		

class Player:
	def update():
		keys = pygame.key.get_pressed()
		if keys[pygame.key.K_RIGHT]:
			self.x += 5
		if keys[pygame.key.K_LEFT]:
			self.x -= 5
		if keys[pygame.key.K_UP]:
			self.y += 5
		if keys[pygame.key.K_DOWN]:
			self.y -= 5

	def __init__(self):
		self.image = get_image("data/images/ship.png")
		self.rect = self.image.get_rect()
		self.rect.center = (self.x, self.y)
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
		return 1

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
	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
#	SimpleBullet()
	last_rects = []
	while update_list:
		turn_actions = get_turn_actions()
		if len(update_list) > max_ob:
			max_ob = len(update_list)
		for bullet in update_list:
			bullet.update()
		for bullet in [bullet for bullet in update_list if bullet.to_remove]:
			update_list.remove(bullet)
			if PARTIAL_UPDATE:
				pygame.display.update(bullet.temp_rect)
		if turn_actions >= 2:
			vframe += 1
			for bullet in update_list:
				bullet.draw()
			if PARTIAL_UPDATE:
				pygame.display.update([b.rect for b in update_list] + [b.last_rect for b in update_list])
			else:
				pygame.display.flip()
			screen.fill( (20,20,20) )
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
