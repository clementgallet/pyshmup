#include <Python.h>
#include <Numeric/arrayobject.h>
#include <GL/gl.h>
#include "constants.h"
#include "stdio.h"

#define array_elem(type) *(double *)(array->data + ARRAY_##type*array->strides[0] + j*array->strides[1])
#define array_ml_elem(type) *(double *)(array_ml->data + ARRAY_ML_##type*array_ml->strides[0] + j*array_ml->strides[1])

static PyObject *draw(PyObject *self, PyObject *args)
{
	int card,card_ml,j;
	PyArrayObject *array;
	PyArrayObject *array_ml;
	PyArg_ParseTuple(args,"O!iO!i",&PyArray_Type,&array,&card,&PyArray_Type,&array_ml,&card_ml);
	
	for (j=0;j<card;j++)
	  {
	    glPushMatrix();
	    glColor4f(1.0, 1.0, 1.0, 0.2);
	    glTranslatef(array_elem(X),array_elem(Y), array_elem(Z));
		 //if ((int) array_elem(UNTIL) > 10000)
		//	 glCallList(3);
		 //else
			 glCallList((int) array_elem(LIST));
	    glPopMatrix();
	  }
	
	for (j=0;j<card_ml;j++)
	  {
	    glPushMatrix();
	    glColor4f(1.0, 1.0, 1.0, 0.9);
	    glTranslatef(array_ml_elem(X),array_ml_elem(Y), array_ml_elem(Z));
	    glCallList((int) array_ml_elem(LIST));
	    glPopMatrix();
	  }
	
	return Py_INCREF(Py_None), Py_None;
}

static PyMethodDef DrawMethods[] = {
	{"draw", draw, METH_VARARGS, "Draw something."},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initdraw(void)
{
	(void) Py_InitModule("draw", DrawMethods);
	import_array();
}
