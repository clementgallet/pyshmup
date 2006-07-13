# -*- coding: utf-8 -*-
# vim: ts=3:sw=3:noet

from constants import *

import pygame
import logging
import math
import time
import OpenGL.GL as gl   # This SUCKS, but PyOpenGL namespaces are broken IMO
import OpenGL.GLU as glu # Macros may save the day

import context

#import hotshot
import sys

from pprint import pprint


############
## Logging

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

l = logging.getLogger('main')
l.setLevel(logging.DEBUG)

############
## Graphics

def set_perspective(width, height):
	global screen
	screen = pygame.display.set_mode( (width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, 24)
	gl.glClear(gl.GL_ACCUM_BUFFER_BIT)

	gl.glViewport(0, 0, width, height)

	gl.glMatrixMode(gl.GL_PROJECTION)
	gl.glLoadIdentity()
	fov = 30
	dist = SCALE*math.tan(math.pi/8) / math.tan(fov*math.pi/360)
	glu.gluPerspective(fov, float(WIDTH)/HEIGHT, dist*0.5, dist*1.5)
	gl.glMatrixMode(gl.GL_MODELVIEW)
	gl.glLoadIdentity()
	gl.glTranslate(0, 0, -dist)

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

	pygame.display.gl_set_attribute(pygame.GL_ACCUM_RED_SIZE, 8)
	pygame.display.gl_set_attribute(pygame.GL_ACCUM_GREEN_SIZE, 8)
	pygame.display.gl_set_attribute(pygame.GL_ACCUM_BLUE_SIZE, 8)
	pygame.display.gl_set_attribute(pygame.GL_ACCUM_ALPHA_SIZE, 8)


	global screen
	screen = pygame.display.set_mode( (WIDTH,HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE )

	set_perspective(WIDTH, HEIGHT)

	gl.glClearAccum(0,0,0,0)
	gl.glClear(gl.GL_ACCUM_BUFFER_BIT)

	gl.glClearColor(*BACKGROUND_COLOR)
	gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)#MINUS_SRC_ALPHA)
	gl.glEnable(gl.GL_BLEND)

	gl.glEnable(gl.GL_TEXTURE_2D)

	pygame.display.set_caption("FIVE TONS OF FLAX !")
	



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
	elif fskip < MIN_SKIP:
		next_ticks = new_ticks + 1000/FPS
		fskip += 1
		return 2	
	else:
		skip_c += 1
		fskip += 1
		next_ticks += 1000/FPS
		return 1

def handle_events(system_state):
	global bullet_list, quit
	for ev in pygame.event.get():
		if ev.type == pygame.VIDEORESIZE:
			set_perspective(ev.w, ev.h)
		if ev.type == pygame.QUIT:
			quit = True
	system_state.keys = pygame.key.get_pressed()

class SystemState(object):
	pass

skip_c = 0
def main():
	global update_list,bullet_list
	init_sdl()
	x=0
	first_ticks = pygame.time.get_ticks()
	frame = 0
	vframe = 0
	max_ob = 0
	first_fps = first_ticks
	frame_loc = 0
#	while bullet_list or foe_list:
	game_context = context.GameContext()
	game_context.load_stage(STAGE_FILE)
	system_state = SystemState()
	while pygame.time.get_ticks() < 50000 and not game_context.done:

		if pygame.time.get_ticks() - first_fps >= 1000:
			pygame.display.set_caption("FIVE TONS OF FLAX ! - fps : " + str(frame_loc))
			first_fps = pygame.time.get_ticks()
			frame_loc = 0

		# Updates system state
		handle_events(system_state)

		# Calculates next game state
		game_context.update(system_state)

		turn_actions = get_turn_actions()
		if len(game_context.update_list) > max_ob:
			max_ob = len(game_context.update_list)

		if turn_actions >= 2:
			frame_loc += 1
			vframe += 1
			game_context.draw()
		frame += 1
		
	l.info("Gameplay  fps : " + str(float(frame)/(pygame.time.get_ticks()-first_ticks)*1000))
	l.info("Graphical fps : " + str(float(vframe)/(pygame.time.get_ticks()-first_ticks)*1000))
	l.info( str(skip_c) + " skips")
	l.info( "max objects # : " + str(max_ob))

#profile.run('main()','prof.txt')

if __name__ == '__main__':
	try:
		hotshot.Profile("prof").runcall(main)
	except NameError:
		main()
