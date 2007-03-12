
import OpenGL.GL as gl

import fonts
import tex

class StaticText(object):
	def __init__(self, text, x, y, frame, font=None):
		self.text = text
		self.x = x
		self.y = y
		self.frame = frame
		self.font = font

	def spawn(self, context):
		self._context = context

		if self.font:
			surf = self.font.render(self.text, True, (255,0,0))
		else:
			surf = fonts.default.render(self.text, True, (255,0,0))
		self.txtw = surf.get_width()
		self.txth = surf.get_height()

		self.tex, tw, th = tex.make_texture(surf)
		self.tw = float(self.txtw)/tw
		self.th = float(self.txth)/th

		context.others_list.append(self)

	def draw(self):
		gl.glEnable(gl.GL_TEXTURE_2D)
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
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
