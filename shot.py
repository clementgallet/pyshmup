
class Shot:

	def __init__(self):

		self.x = 0
		self.y = 0
		self.lines = []

		shot_list.append(self)
		update_list.append(self)
		
		self.to_remove = False
		self.aimed_foe = None
		

		
	def update(self):

		if not self.aimed_foe in foe_list and not self.to_remove:
			self.vanish()
			return self

		dist = (self.x - self.aimed_foe.x)**2 + (self.y - self.aimed_foe.y)**2

		xpos = ((self.aimed_foe.x > self.x) + (self.aimed_foe.x >= self.x) - 1)*(((self.aimed_foe.x - self.x)**2)/dist) + 1
		xneg = 2 - xpos
		ypos = ((self.aimed_foe.y > self.y) + (self.aimed_foe.y >= self.y) - 1)*(((self.aimed_foe.y - self.y)**2)/dist) + 1
		yneg = 2 - ypos

		xpos *= xpos
		xneg *= xneg
		ypos *= ypos
		yneg *= yneg
		
		choix = random.random()*(xpos + xneg + ypos + yneg)

		shot_dist = math.sqrt(dist)/2
		
		if choix < xpos:
			self.x += shot_dist
		elif xpos <= choix < xneg + xpos:
			self.x -= shot_dist
		elif xneg + xpos <= choix < xneg + xpos + ypos:
			self.y += shot_dist
		else:
			self.y -= shot_dist

		self.lines.append((self.x,self.y))

		return self
	
	def draw(self):

		if len(self.lines) > NB_LINES:
			self.lines.pop(0)
			
		taille = len(self.lines) - 1
		c = 0

		if taille >= 0:
			glDisable(GL_TEXTURE_2D)
			for i in self.lines:
				(x,y) = i
				try:
					pen_x = pen_x
					SHOT_COLOR[3] = 1 - (float(taille - c) / NB_LINES)
					glBegin(GL_LINES)
					glColor4f(*SHOT_COLOR)
					glVertex2f(x, y)
					glVertex2f(pen_x, pen_y)
					glEnd()
				except:
					pass
				
				pen_x = x
				pen_y = y
				
				c += 1
			glColor4f(1.0, 1.0, 1.0, 1.0)


	def vanish(self):
		shot_list.remove(self)
		self.to_remove = True
