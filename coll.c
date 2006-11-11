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

	int i;
	int size;

	PyArrayObject *array;
	PyObject *players;
	PyObject *attr;
	PyArg_ParseTuple(args,"O!iOi",&PyArray_Type,&array,&size,&players,&nb_players);

	for (i=0;i < size;i++)
	{
		if (array_elem(UNTIL) >= 0)
			continue;

		array_elem(UNTIL) = NEVER;

		b_x = array_elem(X);
		b_y = array_elem(Y);
		b_d = array_elem(DIRECTION);
		b_s = array_elem(SPEED);

		//printf("b_x = %f / b_y = %f\n",b_x,b_y); 
		/* we build t_x and t_y, lower bounds on the time
		 * till we get a collision between the bullet and
		 * the player (we assume the player is heading dead on)
		 */

		
		for (p_num=0;p_num<nb_players;p_num++)
		{
			attr = PyObject_GetAttrString(PyList_GetItem(players,p_num),"x");
			p_x = PyFloat_AsDouble(attr);
			Py_DECREF(attr);
			attr = PyObject_GetAttrString(PyList_GetItem(players,p_num),"y");
			p_y = PyFloat_AsDouble(attr);
			Py_DECREF(attr);
			attr = PyObject_GetAttrString(PyList_GetItem(players,p_num),"speed");
			p_s = PyFloat_AsDouble(attr);
			Py_DECREF(attr);
			e_x = (b_x > p_x) ? 1 : (b_x < p_x) ? -1 : 0;
			e_y = (b_y > p_y) ? 1 : (b_y < p_y) ? -1 : 0;

			rel_s_x = sin(b_d)*b_s - e_x*p_s;
			rel_s_y = (-1)*cos(b_d)*b_s - e_y*p_s;

			if (dabs(b_x - p_x) < RADIUS)
				t_x = 0;
			else
				t_x = (e_x*RADIUS - b_x + p_x)/rel_s_x;

			if (dabs(b_y - p_y) < RADIUS)
				t_y = 0;
			else
				t_y = (e_y*RADIUS - b_y + p_y)/rel_s_y;

			if ((t_x >= 0) && (t_y >= 0)) /* collision is possible */
			{
				t = max(t_x, t_y);
				array_elem(UNTIL) = min(array_elem(UNTIL), t); /* keeps the minimum time among all the players
				                                                * for a possible collision */
				if (t == 0) /* collision ! */
				{
					array_elem(COLLIDE_MASK) = ((int) array_elem(COLLIDE_MASK)) | (1 << p_num);
					continue;
				}
			}
			/* non collision */
			array_elem(COLLIDE_MASK) = ((int) array_elem(COLLIDE_MASK)) & (-1 - (1 << p_num));

		}
	}
	return Py_INCREF(Py_None), Py_None;
}

static PyObject *update_collisions_ml(PyObject *self, PyObject *args)
{
	double p_x, p_y; /* player coordinates */
	double b_x, b_y; /* bullet coordinates */
	int p_num,nb_players;

	int i,size;

	PyArrayObject *array;
	PyObject *players;
	PyObject *attr;
	PyArg_ParseTuple(args,"O!iOi",&PyArray_Type,&array,&size,&players,&nb_players);

	for (p_num=0;p_num<nb_players;p_num++)
	{
		attr = PyObject_GetAttrString(PyList_GetItem(players,p_num),"x");
		p_x = PyFloat_AsDouble(attr);
		Py_DECREF(attr);
		attr = PyObject_GetAttrString(PyList_GetItem(players,p_num),"y");
		p_y = PyFloat_AsDouble(attr);
		Py_DECREF(attr);
		for (i=0;i<size;i++)
		{
			b_x = array_elem(ML_X);
			b_y = array_elem(ML_Y);

			if (max(dabs(b_x - p_x),dabs(b_y - p_y)) < RADIUS) /* collision ! */
				array_elem(ML_COLLIDE_MASK) = ((int) array_elem(ML_COLLIDE_MASK)) | (1 << p_num);
			else
				array_elem(ML_COLLIDE_MASK) = ((int) array_elem(ML_COLLIDE_MASK)) & (-1 - (1 << p_num));
		}
	}
	return Py_INCREF(Py_None), Py_None;
}

static PyMethodDef DrawMethods[] = {
	{"coll", update_collisions, METH_VARARGS, "Search for collisions"},
	{"collml", update_collisions_ml, METH_VARARGS, "Search for collisions in ml bullets"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcoll(void)
{
	(void) Py_InitModule("coll", DrawMethods);
	import_array();
}
