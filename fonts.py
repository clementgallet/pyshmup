
import pygame

sys_fonts = {}

def init():
	global default
	default = pygame.font.SysFont('', 30)

def get_sys_font(name, size):
	if (name, size) in sys_fonts:
		return sys_fonts[(name,size)]
	else:
		f = pygame.font.SysFont(name, size)
		sys_fonts[(name,size)] = f
		return f
