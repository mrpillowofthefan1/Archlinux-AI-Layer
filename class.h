#ifndef NEW_H
#define NEW_H

void * new      ( const void * class, ... );
void delete     ( void * item );
void * cpy      ( const void * self );
char * print    ( const void * self );

#endif