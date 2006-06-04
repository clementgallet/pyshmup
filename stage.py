import logging

if __name__ == '__main__':
	console = logging.StreamHandler( )
	console.setLevel( logging.DEBUG )
	formatter = logging.Formatter( '%(name)-12s: %(levelname)-8s %(message)s' )
	console.setFormatter( formatter )
	logging.getLogger('').addHandler( console )

l = logging.getLogger('stage')
l.setLevel( logging.DEBUG )

class FoeInfo(object):

	behav = None
	x = 0
	y = 0
	frame = 0
	bullet = None
	sprite = None

class StagetoFoeList:

	root = None
	foe_list = []

	def __init__(self,FILE):

		self.file = FILE
		from xml.dom.minidom import parse
		self.doc = parse(self.file)
		self.root = self.doc.documentElement

	def getFoes(self):

		#if self.foe_list is not None:
		#	return self.foe_list

		#self.foe_list = []

		for foe in self.root.getElementsByTagName("foe"):

			#if foe.nodeType == foe.ELEMENT_NODE:
				
			f = FoeInfo()

			for xml in ['behav','bullet','sprite']:

				try:
					f.__setattr__(xml,foe.getElementsByTagName(xml)[0].childNodes[0].nodeValue)
				except:
					l.warning(xml + ' missing')

			for value in ['x','y','frame']:

				try:
					f.__setattr__(value,eval(foe.getElementsByTagName(value)[0].childNodes[0].nodeValue))
				except:
					l.warning(value + ' missing')

			self.foe_list.append(f)
			
		return self.foe_list
