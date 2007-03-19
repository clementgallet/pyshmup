
import OpenGL.GL as gl

import fonts
import math
import tex

class StaticText(object):
	r = 1.
	g = 1.
	b = 1.
	a = 1.

	def __init__(self, text, x, y, frame, font=None):
		self.text = text
		self.x = x
		self.y = y
		self.frame = frame
		if font is not None:
			self.font = font
		else:
			self.font = fonts.default
		
	def die(self):
		self._context.others_list.remove(self)
		gl.glDeleteTextures(self.tex)

	def spawn(self, context):
		self._context = context

		if self.font:
			surf = self.font.render(self.text, True, (255,255,255))
		else:
			surf = fonts.default.render(self.text, True, (255,255,255))
		self.txtw = surf.get_width()
		self.txth = surf.get_height()

		self.tex, tw, th = tex.make_texture(surf)
		self.tw = float(self.txtw)/tw
		self.th = float(self.txth)/th

		context.others_list.append(self)

	def draw(self):
		gl.glEnable(gl.GL_TEXTURE_2D)
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
		gl.glColor4f(self.r, self.g, self.b, self.a)
		gl.glBegin(gl.GL_QUADS)

		gl.glTexCoord2f(0, 0)
		gl.glVertex2f(self.x, self.y)
		gl.glTexCoord2f(0, self.th)
		gl.glVertex2f(self.x, self.y+self.txth)
		gl.glTexCoord2f(self.tw, self.th)
		gl.glVertex2f(self.x+self.txtw, self.y+self.txth)
		gl.glTexCoord2f(self.tw, 0)
		gl.glVertex2f(self.x+self.txtw, self.y)

		gl.glEnd()
		gl.glColor3i(1,1,1)

class MovingText(StaticText):
	to_remove = False
	k = 120

	def __init__(self, *kargs, **kwargs):
		super(MovingText, self).__init__(*kargs, **kwargs)

	def update(self):
		self.y += 0.2
		self.a = 1-math.exp(-self.k/50.)
		self.k -= 1
		if self.a < 0.01:
			self.to_remove = True
			self.die()
		return self

	def draw(self):
		super(MovingText, self).draw()

	def spawn(self, context):
		super(MovingText, self).spawn(context)

		context.update_list.append(self)
