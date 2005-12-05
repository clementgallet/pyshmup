# -*- coding: utf-8 -*-

from bulletparse import GameObjectController

NO_GRAPHICS = True

import pygame
import logging
import math
import time


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
	screen = pygame.display.set_mode( (320,240), pygame.DOUBLEBUF | pygame.HWSURFACE )
	pygame.display.set_caption("out of funny title ideas...")
	
		
																						

update_list = []

class SimpleBullet:
	direction = 0.0
	speed = 1.0
	x = 160.0
	y = 100.0

	def __init__(self):
		self.controller = GameObjectController()
		self.controller.game_object = self
		self.controller.set_behavior('bee.xml')

		self.image = get_image('data/images/shot.png')
		self.rect = self.image.get_rect()
		self.rect.topleft = (self.x, self.y)

		update_list.append(self)
		self.to_remove = False


	def fire(self, controller, direction=0, speed=1.0):
		#print "launching ", bullet_control, " in hyperspace"
		new_bullet = SimpleBullet()
		new_bullet.controller = controller 
		controller.game_object = new_bullet
		new_bullet.x = self.x
		new_bullet.y = self.y
		new_bullet.direction = direction
		new_bullet.speed = speed

	def update(self):
		if self.x < -20 or self.x > 340 or self.y < -20 or self.y > 260:
			self.to_remove = True
		self.controller.run()
		self.x += math.cos(self.direction*math.pi/180)*self.speed*0.5
		self.y -= math.sin(self.direction*math.pi/180)*self.speed*0.5
		self.rect.center = (self.x, self.y)

	def draw(self):
		screen.blit(self.image, self.rect)
		pass
		
		

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



if __name__ == '__main__':
	init_sdl()
	x=0
	first_ticks = pygame.time.get_ticks()
	frame = 0
	SimpleBullet()
	SimpleBullet()
	SimpleBullet()
	SimpleBullet()
	SimpleBullet()
	SimpleBullet()
	SimpleBullet()
	SimpleBullet()
	while update_list:
		for bullet in update_list:
			bullet.update()
		for bullet in update_list:
			if bullet.to_remove:
				update_list.remove(bullet)
		for bullet in update_list:
			bullet.draw()
		frame += 1
		
		#pygame.display.update(pygame.Rect(200,200,100,100))
		pygame.display.flip()
		screen.fill( (20,20,20) )
		#time.sleep(.02)
		#print bullet.x, bullet.y
	print float(frame)/(pygame.time.get_ticks()-first_ticks)*1000
