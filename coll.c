#include <Python.h>
#include <Numeric/arrayobject.h>
#include <math.h>
#include <stdio.h>

#include "constants.h"



inline double dabs(double x)
{
	return ((x>0) ? x : (-1)*x);
}

inline double max(double x,double y)
{
	return ((x>y) ? x : y);
}

inline double min(double x,double y)
{
	return ((x<y) ? x : y);
}

#define array_elem(type) (*(double *)(array->data + ARRAY_##type*array->strides[0] + i*array->strides[1]))

/**  Calculate a lower bound on the time before collision 
 *  why a player for each bullet.
 *   We assume each player tries his best to die.
 *   Incidentally, detect collisions.
 */
static PyObject *update_collisions(PyObject *self, PyObject *args)
{
	double p_x, p_y; /* player coordinates */
	double b_x, b_y; /* bullet coordinates */
	double p_s; /* player speed */
	double b_s, b_d; /* bullet speed and direction */
	double rel_s_x, rel_s_y; /* relative speeds (bullet - player) */
	int    e_x, e_y; /* approach direction :
	                      1  : player goes positive  ---->
								 0  : player does not move    |
								 -1 : player goes negative  <----
							  (the player tries to hit the bullet) */
	double t_x,t_y;  /* time to hit (on each axis) */
	double t;        /* time to hit (total) */

	int p_num,nb_players;

	int i,size;

	PyArrayObject *array;
	PyListObject *players;
	PyArg_ParseTuple(args,"O!iO!i",&PyArray_Type,&array,&size,&PyList_Type,&players,&nb_players);

	for (p_num=0;p_num<nb_players;p_num++)
	{
		p_x = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GetItem(players,p_num),"x"));
		p_y = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GetItem(players,p_num),"y"));
		p_s = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GetItem(players,p_num),"speed"));
		for (i=0;i<size;i++)
		{
			if (array_elem(UNTIL) > 0)
			{
				array_elem(UNTIL)--;
				continue;
			}

			array_elem(UNTIL) = NEVER;

			b_x = array_elem(X);
			b_y = array_elem(Y);
			b_d = array_elem(DIRECTION);
			b_s = array_elem(SPEED);

			/* we build t_x and t_y, lower bounds on the time
			 * till we get a collision between the bullet and
			 * the player (we assume the player is heading dead on)
			 */

			/* for p_x, p_y, ..... */
		
			e_x = (b_x > p_x) ? 1 : (b_x < p_x) ? -1 : 0;
			e_y = (b_y > p_y) ? 1 : (b_y < p_y) ? -1 : 0;

			rel_s_x = sin(b_d)*b_s - e_x*p_s;
			rel_s_y = -1*cos(b_d)*b_s - e_y*p_s;

			if (dabs(b_x - p_x) < RADIUS)
				t_x = 0;
			else
				t_x = ((b_x - p_x) - e_x*RADIUS)/rel_s_x;

			if (dabs(b_y - p_y) < RADIUS)
				t_y = 0;
			else
				t_y = ((b_y - p_y) - e_y*RADIUS)/rel_s_y;

			if (t_x < 0) /* divergent trajectories */
			{
				if (t_y < 0) /* doubly so */
					t = NEVER;
				else
					t = t_y;
			}
			else
				t = max(t_x, t_y);

			array_elem(UNTIL) = min(array_elem(UNTIL), t);

			if (t == 0) /* collision ! */
				array_elem(COLLIDE_MASK) = ((int) array_elem(COLLIDE_MASK)) | (1 << p_num);
			else
				array_elem(COLLIDE_MASK) = ((int) array_elem(COLLIDE_MASK)) & (-1 - 1 << p_num);
		}
	}
return;
}

static PyMethodDef DrawMethods[] = {
	{"coll", update_collisions, METH_VARARGS, "Search for collisions"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcoll(void)
{
	int i;
	(void) Py_InitModule("coll", DrawMethods);
	import_array();
}
