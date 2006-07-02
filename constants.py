## This file allows definition of constants shared
## between python code and C extensions.

## It may contain only definition of the form :
##   A = value
## where value is a cross-language expression,
## and single line comments starting with a hash sign.
## Comments beginning with two hashes won't be copied to
## the C header.
## `import' statements are ignored.

## Keep in mind assignations will become #defines --
## this means you can use basic arithmetic, but remember
## to parenthetize sensibly.

import pygame

# Players' "hit-disc" radius, in game units
RADIUS = 3.0
# Foes' "hit-disc" radius, in game units
FOE_RADIUS = 10.0
# Player speed, in game units per frame
PLAYER_SPEED = 2.0

# Initial screen dimensions, in pixels
WIDTH = 640
HEIGHT = 480

# Screen height, in game units
SCALE = 400 

# Out-of-screen is e.g. x > width*(1+OUT_LIMIT)
OUT_LIMIT = 0.2 

# BulletML difficulty, 0 to 1
RANK = 0.5

# Pathes
BITMAP_PATH   = "data/images/"
BEHAV_PATH    = "data/bullets/"
SHIP_BITMAP   = "data/images/ship.png"
BULLET_BITMAP = "data/images/shot3.png"
FOE_BITMAP    = "data/images/foe.png"
STAGE_FILE    = "stage.xml"


# Numeric array rows
ARRAY_X = 0
ARRAY_Y = 1
ARRAY_Z = 2
ARRAY_DIRECTION = 3
ARRAY_SPEED = 4
ARRAY_LIST = 5
ARRAY_STATE = 6
ARRAY_UNTIL = 7
ARRAY_LIST_INDEX = 8
ARRAY_OUT_TIME = 9
ARRAY_DIM = 10

# Bullet dangerousness levels
ARRAY_STATE_ML = 0      # fairly unpredictable
ARRAY_STATE_NO_DANG = 1
ARRAY_STATE_DANG = 2




####################
# Derived constants

# 
UNIT_HEIGHT = (SCALE/2)
UNIT_WIDTH = ((UNIT_HEIGHT * WIDTH) / HEIGHT)




#FIXME: Undocumented yet
FOE_LIFE = 30
NB_LINES = 5 # number of lines for the shot





#############################
## END_OF_PORTABLE_CONSTANTS
#############################



# Keys
KEY_SHOT = pygame.K_q

DRAW_HITBOX = True

FONKY_LINES = False

NO_DEATH = True

FPS = 60
MAX_SKIP = 9
MIN_SKIP = 1

BACKGROUND_COLOR = (.235, .275, .275, 1)
BACKGROUND_COLOR = (0, 0, 0, 1)

SHOT_COLOR = [.425, .475, .475, 1]
SHOT_WIDTH = 50

#####################
## Derived constants

if FONKY_LINES:
	LINE_RADIUS2 = 10000*RADIUS2
	FONKY_COLOR = [.205, .245, .245, 1]
	FONKY_OFFSET = [ BACKGROUND_COLOR[i] - FONKY_COLOR[i] for i in [0,1,2] ]
