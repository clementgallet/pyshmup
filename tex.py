
import pygame
import OpenGL.GL as gl

def make_texture(im):
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

	tex = gl.glGenTextures(1)

	gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
	gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
	gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, k_width, 
	           k_height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
	gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
	gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

	return tex, k_width, k_height
