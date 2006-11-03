#include <Python.h>
#include <Numeric/arrayobject.h>
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

#define array_elem(type) (*(double *)(array->data + ARRAY_ML_##type*array->strides[0] + i*array->strides[1]))

/**  Calculate a lower bound on the time before collision 
 *  why a player for each bullet.
 *   We assume each player tries his best to die.
 *   Incidentally, detect collisions.
 */
static PyObject *update_collisions(PyObject *self, PyObject *args)
{
	double p_x, p_y; /* player coordinates */
	double b_x, b_y; /* bullet coordinates */

	int p_num,nb_players;

	int i,size;

	PyArrayObject *array;
	PyObject *players;
	PyArg_ParseTuple(args,"O!iOi",&PyArray_Type,&array,&size,&players,&nb_players);

	for (p_num=0;p_num<nb_players;p_num++)
	{
		p_x = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GetItem(players,p_num),"x"));
		p_y = PyFloat_AsDouble(PyObject_GetAttrString(PyList_GetItem(players,p_num),"y"));
		for (i=0;i<size;i++)
		{
			b_x = array_elem(X);
			b_y = array_elem(Y);

			if (max(dabs(b_x - p_x),dabs(b_y - p_y)) < RADIUS) /* collision ! */
				array_elem(COLLIDE_MASK) = ((int) array_elem(COLLIDE_MASK)) | (1 << p_num);
			else
				array_elem(COLLIDE_MASK) = ((int) array_elem(COLLIDE_MASK)) & (-1 - (1 << p_num));
		}
	}
return Py_INCREF(Py_None), Py_None;
}

static PyMethodDef DrawMethods[] = {
	{"collml", update_collisions, METH_VARARGS, "Search for collisions"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcollml(void)
{
	(void) Py_InitModule("collml", DrawMethods);
	import_array();
}
