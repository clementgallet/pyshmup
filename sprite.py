import logging
import pygame
from OpenGL.GL import *

###########
## Logging

l = logging.getLogger('sprite')

######################
## Sprites management

# sprite cache
sprites = {}
textures = {}

NULL_IMAGE_PATH = 'data/images/not_found.png'

def get_sprite(filename):
	"""Sprite factory : call with the name of an image
	description file name to get an appropriate sprite.
	(Sprites are cached.)"""

	if filename in sprites:
		return sprites[filename]

	# read xml file 'filename'

	try:
		im = pygame.image.load(filename)
	except pygame.error, err:
		if filename == NULL_IMAGE_PATH:
			l.error("Could not find fallback image !")
			return None
		l.error("Could not find sprite image : " + str(filename) )
		l.error("Reason: " + str(err))
		sprites[filename] = get_sprite(NULL_IMAGE_PATH)
		return sprites[filename]

	width = im.get_width()
	k_width = 1
	while k_width < width:
		k_width *= 2

	height = im.get_height()
	k_height = 1
	while k_height < height:
		k_height *= 2

	s = pygame.Surface((k_width, k_height), pygame.SRCALPHA, im)
	s.blit(im, (0,k_height-height))

	data = pygame.image.tostring(s, "RGBA", True)

	bytes = [str(ord(x)) for x in list(data)]
	blocs = [bytes[4*k:4*k+4] for k in xrange(len(bytes)/4)]

	textures[filename] = glGenTextures(1)
	l.info("Loaded texture : %s" % filename)

	glBindTexture(GL_TEXTURE_2D, textures[filename])
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, k_width, 
	        k_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

	#FIXME: get origin coords
	x = im.get_width()/2
	y = im.get_height()/2
	
	# give it to SimpleSprite with coords
	# or create a SpriteGroup that will handle several sprites as one
	# (in case the image won't fit in one texture)

	sprites[filename] = SimpleSprite(tex_id = textures[filename], orig_x = x, \
	            orig_y = y, spr_w = width, spr_h = height, \
					tex_w = float(width)/k_width, \
					tex_h = float(height)/k_height)
	
	return sprites[filename]

class SimpleSprite:
	def __init__(self, tex_id, orig_x, orig_y, tex_w, tex_h, spr_w, spr_h):
		self.tex_id = tex_id
		self.orig_x = orig_x
		self.orig_y = orig_y
		self.tex_w = tex_w
		self.tex_h = tex_h
		self.spr_w = spr_w
		self.spr_h = spr_h

		i = 1
		while glIsList(i):
			i += 1
		self.list = i

		glNewList(self.list, GL_COMPILE)

		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, self.tex_id)
		glBegin(GL_QUADS)

		glTexCoord2f(0, 0)
		glVertex2i(-self.orig_x, -self.orig_y)
		glTexCoord2f(0, self.tex_h)
		glVertex2i(-self.orig_x, self.spr_h-self.orig_y)
		glTexCoord2f(self.tex_w, self.tex_h)
		glVertex2i(self.spr_w-self.orig_x, self.spr_h-self.orig_y)
		glTexCoord2f(self.tex_w, 0)
		glVertex2i(self.spr_w-self.orig_x, -self.orig_y)

		glEnd()

		glEndList()

	def draw(self):
		glCallList(self.list)
