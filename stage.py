import logging

if __name__ == '__main__':
	console = logging.StreamHandler( )
	console.setLevel( logging.DEBUG )
	formatter = logging.Formatter( '%(name)-12s: %(levelname)-8s %(message)s' )
	console.setFormatter( formatter )
	logging.getLogger('').addHandler( console )

l = logging.getLogger('stage')
l.setLevel( logging.DEBUG )

class FoeInfo:

    behav = None
    x = 0
    y = 0
    frame = 0
    bullet = None
    sprite = None

class StagetoFoeList:

    root = None
    foe_list = None

    def __init__(self,FILE):

        self.file = FILE
        self.readXml()


    def readXml(self):

        from xml.dom.minidom import parse
        self.doc = parse(self.file)


    def getRootElement(self):

        if self.root == None:

            self.root = self.doc.documentElement

        return self.root


    def getFoes(self):

        if self.foe_list is not None:
            return self.foe_list

        self.foe_list = []

        for foe in self.getRootElement().getElementsByTagName("foe"):

            if foe.nodeType == foe.ELEMENT_NODE:

                f = FoeInfo()

                try:
                    f.behav = foe.getElementsByTagName("behav")[0].childNodes[0].nodeValue
                except:
                    l.warning('behavior missing')
                try:
                    f.x = eval(foe.getElementsByTagName("x")[0].childNodes[0].nodeValue)
                    f.y = eval(foe.getElementsByTagName("y")[0].childNodes[0].nodeValue)
                except:
                    l.warning('coords missing')
                try:
                    f.frame = eval(foe.getElementsByTagName("frame")[0].childNodes[0].nodeValue)
                except:
                    l.warning('frame missing')
                try:
                    f.bullet = foe.getElementsByTagName("bullet")[0].childNodes[0].nodeValue
                except:
                    l.warning('bullet sprite missing')
                try:
                    f.sprite = foe.getElementsByTagName("sprite")[0].childNodes[0].nodeValue
                except:
                    l.warning('foe sprite missing')

                self.foe_list.append(f)

        return self.foe_list
