#ifndef _REGEXP9_H_
#define _REGEXP9_H_ 1

/*********
* utf.h *
*********/

typedef unsigned short Rune;

typedef struct Resub		Resub;
typedef struct Reclass		Reclass;
typedef struct Reinst		Reinst;
typedef struct Reprog		Reprog;
typedef struct Reljunk		Reljunk;
typedef struct Relist		Relist;

enum
{
    UTFmax      = 3,        /* maximum bytes per rune */
    Runesync    = 0x80,     /* cannot represent part of a UTF sequence (<) */
    Runeself    = 0x80,     /* rune and UTF sequences are the same (<) */
    Runeerror   = 0xFFFD        /* decoding error in UTF */
};

/*************
* regexp9.h *
*************/

#if defined(__cplusplus)
extern "C" { 
#endif

#ifdef AUTOLIB
    AUTOLIB(regexp9)
#endif

        /*
        *	Sub expression matches
        */
    struct Resub{
        union
        {
            char *sp;
            Rune *rsp;
        }s;
        union
        {
            char *ep;
            Rune *rep;
        }e;
    };

    /*
    *	character class, each pair of rune's defines a range
    */
    struct Reclass{
        Rune	*end;
        Rune	spans[64];
    };

    /*
    *	Machine instructions
    */
    struct Reinst{
        int	type;
        union	{
            Reclass	*cp;		/* class pointer */
            Rune	r;		/* character */
            int	subid;		/* sub-expression id for RBRA and LBRA */
            Reinst	*right;		/* right child of OR */
        }u1;
        union {	/* regexp relies on these two being in the same union */
            Reinst *left;		/* left child of OR */
            Reinst *next;		/* next instruction for CAT & LBRA */
        }u2;
    };

    /*
    *	Reprogram definition
    */
    struct Reprog{
        Reinst	*startinst;	/* start pc */
        Reclass	classx[16];	/* .data */
        Reinst	firstinst[5];	/* .text */
    };

    extern __host__ Reprog	*regcomp9(char*);
    extern __device__ int	regexec9(const Reprog*, Reljunk*, Resub*, Relist*, Relist*, char*, int, int);

    // CUDA adapted
    extern __host__ Reprog* cu_regcomp9(char*, void*);
    extern __host__ __device__ Reprog* cu_relocate(Reprog *pp, size_t diff, size_t reinst_size);
    extern __host__ size_t cu_get_reinst_size(char* p);

    extern __host__ __device__ int chartorune(Rune& rune, const char *str);
    extern __host__ __device__ Rune* runestrchr(const Rune *s, Rune c);
    extern __host__ __device__ char* utfrune(char *s, long c);

#if defined(__cplusplus)
}
#endif

/*************
* regcomp.h *
*************/

/*
*  substitution list
*/
#define uchar __reuchar
typedef unsigned char uchar;
#define nelem(x) (sizeof(x)/sizeof((x)[0]))

#define NSUBEXP 32
typedef struct Resublist	Resublist;
struct	Resublist
{
    Resub	m[NSUBEXP];
};

/* max character classes per program */
extern Reprog	RePrOg;
#define	NCLASS	(sizeof(RePrOg.classx)/sizeof(Reclass))

/* max rune ranges per character class */
#define NCCRUNE	(sizeof(Reclass)/sizeof(Rune))

/*
* Actions and Tokens (Reinst types)
*
*	02xx are operators, value == precedence
*	03xx are tokens, i.e. operands for operators
*/
#define RUNE		0177
#define	OPERATOR	0200	/* Bitmask of all operators */
#define	START		0200	/* Start, used for marker on stack */
#define	RBRA		0201	/* Right bracket, ) */
#define	LBRA		0202	/* Left bracket, ( */
#define	OR		0203	/* Alternation, | */
#define	CAT		0204	/* Concatentation, implicit operator */
#define	STAR		0205	/* Closure, * */
#define	PLUS		0206	/* a+ == aa* */
#define	QUEST		0207	/* a? == a|nothing, i.e. 0 or 1 a's */
#define	ANY		0300	/* Any character except newline, . */
#define	ANYNL		0301	/* Any character including newline, . */
#define	NOP		0302	/* No operation, internal use only */
#define	BOL		0303	/* Beginning of line, ^ */
#define	EOL		0304	/* End of line, $ */
#define	CCLASS		0305	/* Character class, [] */
#define	NCCLASS		0306	/* Negated character class, [] */
#define	END		0377	/* Terminate: match found */

/*
*  regexec execution lists
*/
#define LISTSIZE	10
#define BIGLISTSIZE	(10*LISTSIZE)
typedef struct Relist	Relist;
struct Relist
{
    Reinst*		inst;	/* Reinstruction of the thread */
    Resublist	se;		/* matched subexpressions in this thread */
};
typedef struct Reljunk	Reljunk;
struct	Reljunk
{
    Relist*	relist[2];
    Relist*	reliste[2];
    int	starttype;
    Rune	startchar;
    char*	starts;
    char*	eol;
    Rune*	rstarts;
    Rune*	reol;
};

#endif
