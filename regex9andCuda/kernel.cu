
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <setjmp.h>
//#include <unistd.h>
#include "regexp9.h"

#define nil 0
#define exits(x) exit(x && *x ? 1 : 0)
#include <stdarg.h>
#include <string.h>


#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <fstream>

/**
 * @def THREADS();
 *
 * @brief   Number of CUDA threads per block
 */
#define THREADS 512

/**
 * @def BLOCKS();
 *
 * @brief   Number of CUDA blocks to run
 */
#define BLOCKS 64

// Temporary data for regexp programs stores in these variables.
// I was tried to move this allocation to host function but it not that simple
// and not so important where to allocate for run and test time.
// Global memory, read & write.
__device__ Relist relist0[BLOCKS*THREADS][LISTSIZE];
__device__ Relist relist1[BLOCKS*THREADS][LISTSIZE];

/**
 * @fn  __host__ __device__ char * cu_strchr(const char *s, int ch)
 *
 * @brief   Standart C strchr function implemented for CUDA.
 *
 * @param   s   The const char * to process.
 * @param   ch  The char to find.
 *
 * @return  null if not found, else first position of ch in const char* s.
 */
__host__ __device__ char *
    cu_strchr(const char *s, int ch)
{
    /* scan from left to right */
    while (*s) {
        /* if we hit it, return it */
        if (*s==ch) {
            return (char *)s;
        }
        s++;
    }

    /* if we were looking for the 0, return that */
    if (*s==ch) {
        return (char *)s;
    }

    /* didn't find it */
    return NULL;
}

/*
Copyright information from the original package:
*/

/*
The authors of this software is Rob Pike.
Copyright (c) 2002 by Lucent Technologies.
Permission to use, copy, modify, and distribute this software for any
purpose without fee is hereby granted, provided that this entire notice
is included in all copies of any software which is or includes a copy
or modification of this software and in all copies of the supporting
documentation for such software.
THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.


This is a Unix port of the Plan 9 regular expression library.

Please send comments about the packaging
to Russ Cox <rsc@swtch.com>.

----

This software is also made available under the Lucent Public License
version 1.02; see http://plan9.bell-labs.com/plan9dist/license.html
*/

/*
This software was packaged for Unix by Russ Cox.
Please send comments to rsc@swtch.com.

http://swtch.com/plan9port/unix
*/

enum
{
    Bit1	= 7,
    Bitx	= 6,
    Bit2	= 5,
    Bit3	= 4,
    Bit4	= 3,

    T1	= ((1<<(Bit1+1))-1) ^ 0xFF,	/* 0000 0000 */
    Tx	= ((1<<(Bitx+1))-1) ^ 0xFF,	/* 1000 0000 */
    T2	= ((1<<(Bit2+1))-1) ^ 0xFF,	/* 1100 0000 */
    T3	= ((1<<(Bit3+1))-1) ^ 0xFF,	/* 1110 0000 */
    T4	= ((1<<(Bit4+1))-1) ^ 0xFF,	/* 1111 0000 */

    Rune1	= (1<<(Bit1+0*Bitx))-1,		/* 0000 0000 0111 1111 */
    Rune2	= (1<<(Bit2+1*Bitx))-1,		/* 0000 0111 1111 1111 */
    Rune3	= (1<<(Bit3+2*Bitx))-1,		/* 1111 1111 1111 1111 */

    Maskx	= (1<<Bitx)-1,			/* 0011 1111 */
    Testx	= Maskx ^ 0xFF,			/* 1100 0000 */

    Bad	= Runeerror
};


__device__ int
    chartorune(Rune& rune, const char *str)
{
    int c, c1, c2;
    long l;

    /*
    * one character sequence
    *	00000-0007F => T1
    */
    c = *(uchar*)str;
    if(c < Tx) {
        rune = c;
        return 1;
    }
    /*
    * two character sequence
    *	0080-07FF => T2 Tx
    */
    c1 = *(uchar*)(str+1) ^ Tx;
    if(c1 & Testx)
        goto bad;
    if(c < T3) {
        if(c < T2)
            goto bad;
        l = ((c << Bitx) | c1) & Rune2;
        if(l <= Rune1)
            goto bad;
        rune = l;
        return 2;
    }

    /*
    * three character sequence
    *	0800-FFFF => T3 Tx Tx
    */
    c2 = *(uchar*)(str+2) ^ Tx;
    if(c2 & Testx)
        goto bad;
    if(c < T4) {
        l = ((((c << Bitx) | c1) << Bitx) | c2) & Rune3;
        if(l <= Rune2)
            goto bad;
        rune = l;
        return 3;
    }

    /*
    * bad decoding
    */
bad:
    rune = Bad;
    return 1;
}

__device__ Rune*
    runestrchr(const Rune *s, Rune c)
{
    Rune c0 = c;
    Rune c1;

    if(c == 0) {
        while(*s++)
            ;
        return (Rune*)s-1;
    }

    while((c1 = *s++))
        if(c1 == c0)
            return (Rune*)s-1;
    return 0;
}

__device__ char*
    utfrune(char *s, long c)
{
    long c1;
    Rune r;
    int n;

    if(c < Runesync)		/* not part of utf sequence */
        return cu_strchr(s, c);

    for(;;) {
        c1 = *(uchar*)s;
        if(c1 < Runeself) {	/* one byte rune */
            if(c1 == 0)
                return 0;
            if(c1 == c)
                return s;
            s++;
            continue;
        }

        n = chartorune(r, s);
        if(r == c)
            return s;
        s += n;
    }
}

/************
* regaux.c *
************/

/*
*  save a new match in mp
*/
__device__ __host__ static void
    _renewmatch(int ms, Resublist& sp, Resub* mp)
{
    int i;

    if(mp==0 || ms<=0)
        return;
    if(mp[0].s.sp==0 || sp.m[0].s.sp<mp[0].s.sp ||
        (sp.m[0].s.sp==mp[0].s.sp && sp.m[0].e.ep>mp[0].e.ep)){
            for(i=0; i<ms && i<NSUBEXP; i++)
                mp[i] = sp.m[i];
            for(; i<ms; i++)
                mp[i].s.sp = mp[i].e.ep = 0;
    }
}

/*
* Note optimization in _renewthread:
* 	*lp must be pending when _renewthread called; if *l has been looked
*		at already, the optimization is a bug.
*/
__device__ __host__ static Relist*
    _renewthread(Relist *lp,	/* _relist to add to */
    Reinst *ip,		            /* instruction to add */
    int ms,
    Resublist& sep)		        /* pointers to subexpressions */
{
    Relist *p;

    for(p=lp; p->inst; p++){
        if(p->inst == ip){
            if(sep.m[0].s.sp < p->se.m[0].s.sp){
                if(ms > 1)
                    p->se = sep;
                else
                    p->se.m[0] = sep.m[0];
            }
            return 0;
        }
    }
    p->inst = ip;
    if(ms > 1)
        p->se = sep;
    else
        p->se.m[0] = sep.m[0];
    (++p)->inst = 0;
    return p;
}

/*
* same as renewthread, but called with
* initial empty start pointer.
*/
__host__ __device__ static Relist*
    _renewemptythread(Relist *lp,	/* _relist to add to */
    Reinst *ip,		/* instruction to add */
    int ms,
    char *sp)		/* pointers to subexpressions */
{
    Relist *p;

    for(p=lp; p->inst; p++){
        if(p->inst == ip){
            if(sp < p->se.m[0].s.sp) {
                if(ms > 1)
                    memset(&p->se, 0, sizeof(p->se));
                p->se.m[0].s.sp = sp;
            }
            return 0;
        }
    }
    p->inst = ip;
    if(ms > 1)
        memset(&p->se, 0, sizeof(p->se));
    p->se.m[0].s.sp = sp;
    (++p)->inst = 0;
    return p;
}

/*************
* regcomp.c *
*************/
#define	TRUE	1
#define	FALSE	0

/*
* Parser Information
*/
typedef
struct Node
{
    Reinst*	first;
    Reinst*	last;
}Node;

#define	NSTACK	20
static	Node	andstack[NSTACK];
static	Node	*andp;
static	int	atorstack[NSTACK];
static	int*	atorp;
static	int	cursubid;		/* id of current subexpression */
static	int	subidstack[NSTACK];	/* parallel to atorstack */
static	int*	subidp;
static	int	lastwasand;	    /* Last token was operand */
static	int	nbra;
static	char*	exprp;		/* pointer to next character in source expression */
static	int	lexdone;
static	int	nclass;
static	Reclass*classp;
static	Reinst*	freep;
static	int	errors;
static	Rune	yyrune;		/* last lex'd rune */
static	Reclass*yyclassp;	/* last lex'd class */

/* predeclared crap */
static	void	operatorx(int);
static	void	pushand(Reinst*, Reinst*);
static	void	pushator(int);
static	void	evaluntil(int);
static	int	    bldcclass(void);

static jmp_buf regkaboom;

__host__ static	void
    rcerror(char *s)
{
    /* saving s is removed for cuda version */
    errors++;
    longjmp(regkaboom, 1);
}

__host__ static	Reinst*
    newinst(int t)
{
    freep->type = t;
    freep->u2.left = 0;
    freep->u1.right = 0;
    return freep++;
}

__host__ static	void
    operand(int t)
{
    Reinst *i;

    if(lastwasand)
        operatorx(CAT);	/* catenate is implicit */
    i = newinst(t);

    if(t == CCLASS || t == NCCLASS)
        i->u1.cp = yyclassp;
    if(t == RUNE)
        i->u1.r = yyrune;

    pushand(i, i);
    lastwasand = TRUE;
}

__host__ static	void
    operatorx(int t)
{
    if(t==RBRA && --nbra<0)
        rcerror("unmatched right paren");
    if(t==LBRA){
        if(++cursubid >= NSUBEXP)
            rcerror ("too many subexpressions");
        nbra++;
        if(lastwasand)
            operatorx(CAT);
    } else
        evaluntil(t);
    if(t != RBRA)
        pushator(t);
    lastwasand = FALSE;
    if(t==STAR || t==QUEST || t==PLUS || t==RBRA)
        lastwasand = TRUE;	/* these look like operands */
}

__host__ static	void
    regerr2(char *s, int c)
{
    char buf[100];
    char *cp = buf;
    while(*s)
        *cp++ = *s++;
    *cp++ = c;
    *cp = '\0'; 
    rcerror(buf);
}

__host__ static	void
    cant(char *s)
{
    char buf[100];
    strcpy(buf, "can't happen: ");
    strcat(buf, s);
    rcerror(buf);
}

__host__ static	void
    pushand(Reinst *f, Reinst *l)
{
    if(andp >= &andstack[NSTACK])
        cant("operand stack overflow");
    andp->first = f;
    andp->last = l;
    andp++;
}

__host__ static	void
    pushator(int t)
{
    if(atorp >= &atorstack[NSTACK])
        cant("operator stack overflow");
    *atorp++ = t;
    *subidp++ = cursubid;
}

__host__ static	Node*
    popand(int op)
{
    Reinst *inst;

    if(andp <= &andstack[0]){
        regerr2("missing operand for ", op);
        inst = newinst(NOP);
        pushand(inst,inst);
    }
    return --andp;
}

__host__ static	int
    popator(void)
{
    if(atorp <= &atorstack[0])
        cant("operator stack underflow");
    --subidp;
    return *--atorp;
}

__host__ static	void
    evaluntil(int pri)
{
    Node *op1, *op2;
    Reinst *inst1, *inst2;

    while(pri==RBRA || atorp[-1]>=pri){
        switch(popator()){
        default:
            rcerror("unknown operator in evaluntil");
            break;
        case LBRA:		/* must have been RBRA */
            op1 = popand('(');
            inst2 = newinst(RBRA);
            inst2->u1.subid = *subidp;
            op1->last->u2.next = inst2;
            inst1 = newinst(LBRA);
            inst1->u1.subid = *subidp;
            inst1->u2.next = op1->first;
            pushand(inst1, inst2);
            return;
        case OR:
            op2 = popand('|');
            op1 = popand('|');
            inst2 = newinst(NOP);
            op2->last->u2.next = inst2;
            op1->last->u2.next = inst2;
            inst1 = newinst(OR);
            inst1->u1.right = op1->first;
            inst1->u2.left = op2->first;
            pushand(inst1, inst2);
            break;
        case CAT:
            op2 = popand(0);
            op1 = popand(0);
            op1->last->u2.next = op2->first;
            pushand(op1->first, op2->last);
            break;
        case STAR:
            op2 = popand('*');
            inst1 = newinst(OR);
            op2->last->u2.next = inst1;
            inst1->u1.right = op2->first;
            pushand(inst1, inst1);
            break;
        case PLUS:
            op2 = popand('+');
            inst1 = newinst(OR);
            op2->last->u2.next = inst1;
            inst1->u1.right = op2->first;
            pushand(op2->first, inst1);
            break;
        case QUEST:
            op2 = popand('?');
            inst1 = newinst(OR);
            inst2 = newinst(NOP);
            inst1->u2.left = inst2;
            inst1->u1.right = op2->first;
            op2->last->u2.next = inst2;
            pushand(inst1, inst2);
            break;
        }
    }
}

__host__ static	Reprog*
    optimize(Reprog *pp)
{
    Reinst *inst, *target;
    int size;
    Reprog *npp;
    Reclass *cl;
    int diff;

    /*
    *  get rid of NOOP chains
    */
    for(inst=pp->firstinst; inst->type!=END; inst++){
        target = inst->u2.next;
        while(target->type == NOP)
            target = target->u2.next;
        inst->u2.next = target;
    }

    /*
    *  The original allocation is for an area larger than
    *  necessary.  Reallocate to the actual space used
    *  and then relocate the code.
    */
    size = sizeof(Reprog) + (freep - pp->firstinst)*sizeof(Reinst);
    npp = (Reprog*) realloc(pp, size);
    if(npp==0 || npp==pp)
        return pp;
    diff = (char *)npp - (char *)pp;
    freep = (Reinst *)((char *)freep + diff);
    for(inst=npp->firstinst; inst<freep; inst++){
        switch(inst->type){
        case OR:
        case STAR:
        case PLUS:
        case QUEST:
            inst->u1.right = (Reinst*)((char*)inst->u1.right + diff);
            break;
        case CCLASS:
        case NCCLASS:
            inst->u1.right = (Reinst*)((char*)inst->u1.right + diff);
            cl = inst->u1.cp;
            cl->end = (Rune*)((char*)cl->end + diff);
            break;
        }
        inst->u2.left = (Reinst*)((char*)inst->u2.left + diff);
    }
    npp->startinst = (Reinst*)((char*)npp->startinst + diff);
    return npp;
}

/**
 * @fn  __host__ Reprog* cu_optimize(Reprog *pp, void* memory)
 *
 * @brief   Optimization of automaton in cuda version should not
 * 			realloc memory block as it does simple optimize function.
 *
 * @param [in,out]  pp      The regexp program pointer.
 *
 * @return The regexp program pointer.
 */

__host__ Reprog*
    cu_optimize(Reprog *pp)
{
    Reinst *inst, *target;

    /*
    *  get rid of NOOP chains
    */
    for(inst=pp->firstinst; inst->type!=END; inst++){
        target = inst->u2.next;
        while(target->type == NOP)
            target = target->u2.next;
        inst->u2.next = target;
    }

    /*
    * Then standart regexp9 optimize function performs following:
    * 
    *  The original allocation is for an area larger than
    *  necessary.  Reallocate to the actual space used
    *  and then relocate the code.
    *  
    *  But in cuda we just should return our pointer.
    */
    return pp;
}

/**
 * @fn  Reprog * cu_relocate(Reprog *pp, size_t diff, size_t reinst_size)
 *
 * @brief   Relocation of regexp program to another memory block, for example
 * 			after we copy it to CUDA kernel, we should fix the pointers.
 *
 * @param [in,out]  pp  The regexp program pointer.
 * @param   diff        The difference between old address and new address.
 * @param   reinst_size Size of the reinst(to find an end of the memory block).
 *
 * @return  The regexp program pointer.
 */

Reprog *
    cu_relocate(Reprog *pp, size_t diff, size_t reinst_size) {
        Reinst *inst;
        Reclass *cl;
        Reinst *mfreep;

        // Last reinst address.
        mfreep = (Reinst *)((char *)pp + reinst_size);
        
        /**
        * Iterating through all reinsts and shift pointers addresses for diff 
        * value.
        */
        for(inst=pp->firstinst; inst<mfreep; inst++){
            switch(inst->type){
            case OR:
            case STAR:
            case PLUS:
            case QUEST:
                inst->u1.right = (Reinst*)((char*)inst->u1.right + diff);
                break;
            case CCLASS:
            case NCCLASS:
                inst->u1.right = (Reinst*)((char*)inst->u1.right + diff);
                cl = inst->u1.cp;
                cl->end = (Rune*)((char*)cl->end + diff);
                break;
            }
            inst->u2.left = (Reinst*)((char*)inst->u2.left + diff);
        }
        // Shifting the startinst pointer as well.
        pp->startinst = (Reinst*)((char*)pp->startinst + diff);
        return pp;
}

__host__ static	Reclass*
    newclass(void)
{
    if(nclass >= NCLASS)
        regerr2("too many character classes; limit", NCLASS+'0');
    return &(classp[nclass++]);
}

__host__ static	int
    nextc(Rune& rp)
{
    if(lexdone){
        rp = 0;
        return 1;
    }
    exprp += chartorune(rp, exprp);
    if(rp == '\\'){
        exprp += chartorune(rp, exprp);
        return 1;
    }
    if(rp == 0)
        lexdone = 1;
    return 0;
}

__host__ static	int
    lex(int literal, int dot_type)
{
    int quoted;

    quoted = nextc(yyrune);
    if(literal || quoted){
        if(yyrune == 0)
            return END;
        return RUNE;
    }

    switch(yyrune){
    case 0:
        return END;
    case '*':
        return STAR;
    case '?':
        return QUEST;
    case '+':
        return PLUS;
    case '|':
        return OR;
    case '.':
        return dot_type;
    case '(':
        return LBRA;
    case ')':
        return RBRA;
    case '^':
        return BOL;
    case '$':
        return EOL;
    case '[':
        return bldcclass();
    }
    return RUNE;
}

__host__ static int
    bldcclass(void)
{
    int type;
    Rune r[NCCRUNE];
    Rune *p, *ep, *np;
    Rune rune;
    int quoted;

    /* we have already seen the '[' */
    type = CCLASS;
    yyclassp = newclass();

    /* look ahead for negation */
    /* SPECIAL CASE!!! negated classes don't match \n */
    ep = r;
    quoted = nextc(rune);
    if(!quoted && rune == '^'){
        type = NCCLASS;
        quoted = nextc(rune);
        *ep++ = '\n';
        *ep++ = '\n';
    }

    /* parse class into a set of spans */
    for(; ep<&r[NCCRUNE];){
        if(rune == 0){
            rcerror("malformed '[]'");
            return 0;
        }
        if(!quoted && rune == ']')
            break;
        if(!quoted && rune == '-'){
            if(ep == r){
                rcerror("malformed '[]'");
                return 0;
            }
            quoted = nextc(rune);
            if((!quoted && rune == ']') || rune == 0){
                rcerror("malformed '[]'");
                return 0;
            }
            *(ep-1) = rune;
        } else {
            *ep++ = rune;
            *ep++ = rune;
        }
        quoted = nextc(rune);
    }

    /* sort on span start */
    for(p = r; p < ep; p += 2){
        for(np = p; np < ep; np += 2)
            if(*np < *p){
                rune = np[0];
                np[0] = p[0];
                p[0] = rune;
                rune = np[1];
                np[1] = p[1];
                p[1] = rune;
            }
    }

    /* merge spans */
    np = yyclassp->spans;
    p = r;
    if(r == ep)
        yyclassp->end = np;
    else {
        np[0] = *p++;
        np[1] = *p++;
        for(; p < ep; p += 2)
            if(p[0] <= np[1]){
                if(p[1] > np[1])
                    np[1] = p[1];
            } else {
                np += 2;
                np[0] = p[0];
                np[1] = p[1];
            }
            yyclassp->end = np+2;
    }

    return type;
}

__host__ static	Reprog*
    regcomp1(char *s, int literal, int dot_type)
{
    int token;
    Reprog *volatile pp;

    /* get memory for the program */
    pp = (Reprog*) malloc(sizeof(Reprog) + 6*sizeof(Reinst)*strlen(s));
    if(pp == 0){
        return 0;
    }
    freep = pp->firstinst;
    classp = pp->classx;
    errors = 0;

    if(setjmp(regkaboom))
        goto out;

    /* go compile the sucker */
    lexdone = 0;
    exprp = s;
    nclass = 0;
    nbra = 0;
    atorp = atorstack;
    andp = andstack;
    subidp = subidstack;
    lastwasand = FALSE;
    cursubid = 0;

    /* Start with a low priority operator to prime parser */
    pushator(START-1);
    while((token = lex(literal, dot_type)) != END){
        if((token&0300) == OPERATOR)
            operatorx(token);
        else
            operand(token);
    }

    /* Close with a low priority operator */
    evaluntil(START);

    /* Force END */
    operand(END);
    evaluntil(START);
#ifdef DEBUG
    dumpstack();
#endif
    if(nbra)
        rcerror("unmatched left paren");
    --andp;	/* points to first and only operand */
    pp->startinst = andp->first;
#ifdef DEBUG
    dump(pp);
#endif
    pp = optimize(pp);
#ifdef DEBUG
    print("start: %d\n", andp->first-pp->firstinst);
    dump(pp);
#endif
out:
    if(errors){
        free(pp);
        pp = 0;
    }
    return pp;
}

/**
 * @fn  static Reprog* cu_regcomp1(char *s, int literal, int dot_type, void* memory)
 *
 * @brief   Compilation of regular expression for CUDA. In contrast to original regcomp1
 * 			we can't shouldn't allocate memory inside. We just pass the void* to block of
 * 			memory, where it should store the data.
 *
 * @param [in]  s           Regular expression.
 * @param   literal         The literal (not used).
 * @param   dot_type        Type of the dot (not used).
 * @param [in]  memory      Pointer to memory block where automaton should be stored.
 *
 * @return Pointer to regexp program. Null if it fails.
 */

static	Reprog*
    cu_regcomp1(char *s, int literal, int dot_type, void* memory)
{
    int token;
    Reprog *volatile pp;

    /* get memory for the program */
    pp = (Reprog*) memory; /*malloc(sizeof(Reprog) + 6*sizeof(Reinst)*strlen(s));*/

    freep = pp->firstinst;
    classp = pp->classx;
    errors = 0;

    if(setjmp(regkaboom))
        goto out;

    /* go compile the sucker */
    lexdone = 0;
    exprp = s;
    nclass = 0;
    nbra = 0;
    atorp = atorstack;
    andp = andstack;
    subidp = subidstack;
    lastwasand = FALSE;
    cursubid = 0;

    /* Start with a low priority operator to prime parser */
    pushator(START-1);
    while((token = lex(literal, dot_type)) != END){
        if((token&0300) == OPERATOR)
            operatorx(token);
        else
            operand(token);
    }

    /* Close with a low priority operator */
    evaluntil(START);

    /* Force END */
    operand(END);
    evaluntil(START);
#ifdef DEBUG
    dumpstack();
#endif
    if(nbra)
        rcerror("unmatched left paren");
    --andp;	/* points to first and only operand */
    pp->startinst = andp->first;
#ifdef DEBUG
    dump(pp);
#endif
    pp = cu_optimize(pp);
#ifdef DEBUG
    print("start: %d\n", andp->first-pp->firstinst);
    dump(pp);
#endif
out:
    if(errors){
        free(pp);
        pp = 0;
    }
    return pp;
}

/**
 * @fn  size_t cu_get_reinst_size(char* p)
 *
 * @brief   Getting reinst size after compilation.
 *
 * @param [in]  p   Address to memory block, where automaton is stored.
 *
 * @return  .
 */

size_t cu_get_reinst_size(char* p) {
    return (char*)freep - (char*)p;
}

__host__ extern	Reprog*
    regcomp9(char *s)
{
    return regcomp1(s, 0, ANY);
}

/**
 * @fn  __host__ extern Reprog* cu_regcomp9(char *s, void* memory)
 *
 * @brief   Public interface function to regexp9 library that compiles
 * 			regexp automaton.
 *
 * @param [in]  s       Regular expression.
 * @param [in]  memory  Pointer to memory block where automaton should be stored.
 *
 * @return  null if it fails, else pointer to regexp program.
 */

__host__ extern	Reprog*
    cu_regcomp9(char *s, void* memory)
{
    return cu_regcomp1(s, 0, ANY, memory);
}

/*************
* regexec.c *
*************/

/**
 * @fn  __host__ __device__ static int regexec1(const Reprog *progp, char *bol, Reljunk* j,
 * int ms, Resub* mp, int cu_i )
 *
 * @brief   Inner function that performs regexp matching. The most interesting to optimize.
 *
 * @param   progp       The regexp program.
 * @param [in] bol      String to run machine on.
 * @param [in] j        Temporary data for automaton.
 * @param   ms          Number of elements at mp. How much matches should we find.
 * @param [in,out]  mp  Resub array where result is stored.
 *
 * @return  0 if no match
 *  		>0 if a match
 * 	    	<0 if we ran out of _relist space
 */

__host__ __device__ static int
    regexec1(const Reprog *progp,
    char *bol,	
    Reljunk* j,
    int ms,		
    Resub* mp
    )
{
    int flag=0;
    Reinst *inst;
    Relist *tlp;
    int checkstart;
    Rune r, *rp, *ep;
    int n;
    Relist* tl;		/* This list, next list */
    Relist* nl;
    Relist* tle;	/* ends of this and next list */
    Relist* nle;
    int match = 0;
    char *p;
    char *s;

    checkstart = j->starttype;

    if(mp)
        for(int i=0; i<ms; i++) {
            mp[i].s.sp = 0;
            mp[i].e.ep = 0;
        }

    j->relist[0][0].inst = 0;	
    j->relist[1][0].inst = 0;

    /* Execute machine once for each character, including terminal NUL */
    s = j->starts;
    do{
        /* fast check for first char */
        if(checkstart) {
            if (j->starttype == RUNE) {
                p = utfrune(s, j->startchar);
                if(p == 0 || s == j->eol)
                    return match;
                s = p;
                break;
            } else {
                if(s == bol)
                    break;
                p = utfrune(s, '\n');
                if(p == 0 || s == j->eol)
                    return match;
                s = p+1;
                break;
            }
        }

        r = *(uchar*)s;

        n = (r < Runeself) ? 1 : chartorune(r, s);

        /* switch run lists */
        tl  = j->relist[flag];
        tle = j->reliste[flag];
        nl  = j->relist[flag^=1];
        nle = j->reliste[flag];
        nl->inst = 0;

        /* Add first instruction to current list */
        if(match == 0)
            _renewemptythread(tl, progp->startinst, ms, s);

        /* Execute machine until current list is empty */
        for(tlp=tl; tlp->inst; tlp++){	/* assignment = */
            for(inst = tlp->inst; ; inst = inst->u2.next){
                switch(inst->type){
                case RUNE:	/* regular character */
                    if(inst->u1.r == r){
                        if(_renewthread(nl, inst->u2.next, ms, tlp->se)==nle)
                            return -1;
                    }
                    break;
                case LBRA:
                    tlp->se.m[inst->u1.subid].s.sp = s;
                    continue;
                case RBRA:
                    tlp->se.m[inst->u1.subid].e.ep = s;
                    continue;
                case ANY:
                    if(r != '\n')
                        if(_renewthread(nl, inst->u2.next, ms, tlp->se)==nle)
                            return -1;
                    break;
                case ANYNL:
                    if(_renewthread(nl, inst->u2.next, ms, tlp->se)==nle)
                        return -1;
                    break;
                case BOL:
                    if(s == bol || *(s-1) == '\n')
                        continue;
                    break;
                case EOL:
                    if(s == j->eol || r == 0 || r == '\n')
                        continue;
                    break;
                case CCLASS:
                    ep = inst->u1.cp->end;
                    for(rp = inst->u1.cp->spans; rp < ep; rp += 2)
                        if(r >= rp[0] && r <= rp[1]){
                            if(_renewthread(nl, inst->u2.next, ms, tlp->se)==nle)
                                return -1;
                            break;
                        }
                        break;
                case NCCLASS:
                    ep = inst->u1.cp->end;
                    for(rp = inst->u1.cp->spans; rp < ep; rp += 2)
                        if(r >= rp[0] && r <= rp[1])
                            break;
                    if(rp == ep)
                        if(_renewthread(nl, inst->u2.next, ms, tlp->se)==nle)
                            return -1;
                    break;
                case OR:
                    /* evaluate right choice later */
                    if(_renewthread(tlp, inst->u1.right, ms, tlp->se) == tle)
                        return -1;
                    /* efficiency: advance and re-evaluate */
                    continue;
                case END:	/* Match! */
                    match = 1;
                    tlp->se.m[0].e.ep = s;
                    if(mp != 0)
                        _renewmatch(ms, tlp->se, mp);
                    break;
                }
                break;
            }
        }
        if(*s == 0)
            break;
        checkstart = j->starttype && nl->inst==0;
        s += n;
    } while(r);
    return match;
}

/**
 * @fn  __device__ extern int regexec9(const Reprog *progp, char *bol, Reljunk* rj, Resub* rs,
 * Relist* dev_relist0, Relist* dev_relist1, int ms, int cu_i)
 *
 * @brief  Public interface function to regexp9 library that prepares and runs
 *         regexp automaton.
 * 			
 * @param   progp       The regexp program.
 * @param [in] bol      String to run machine on.
 * @param [in] rj       Temporary data for automaton.
 * @param [in,out]  rs  If non-null, the rs.
 * @param   ms          Number of elements at mp. How much matches should we find.
 * @param   cu_i        Cuda thread ID.
 *
 * @return  0 if no match
 *  		>0 if a match
 * 	    	<0 if we ran out of _relist space
 */

__device__ extern int
    regexec9(const Reprog *progp,	
    char *bol,	
    Reljunk* rj,
    Resub* rs,
    int ms,		
    int cu_i)		
{
    Resub *mp = rs; /* subexpression elements */
    int rv;
    Reljunk* j = rj;

    /*
    *  use user-specified starting/ending location if specified
    */
    j->starts = bol;
    j->eol = 0;
    if(mp && ms>0){
        if(mp->s.sp)
            j->starts = mp->s.sp;
        if(mp->e.ep)
            j->eol = mp->e.ep;
    }

    j->starttype = 0;
    j->startchar = 0;
    if(progp->startinst->type == RUNE && progp->startinst->u1.r < Runeself) {
        j->starttype = RUNE;
        j->startchar = progp->startinst->u1.r;
    }
    if(progp->startinst->type == BOL)
        j->starttype = BOL;

    /* mark space of relists */
    j->relist[0] = relist0[cu_i];
    j->relist[1] = relist1[cu_i];
    j->reliste[0] = relist0[cu_i] + nelem(relist0[cu_i]) - 2;
    j->reliste[1] = relist1[cu_i] + nelem(relist1[cu_i]) - 2;
    
    /* start magic */
    rv = regexec1(progp, bol, j, ms, mp);
    if(rv >= 0)
        return rv;

    return -1;
}


////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions from SDK

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
            file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}
//
////////////////////////////////////////////////////////////////////////////////


void matchWithCuda(const std::vector<char>& input, 
    const std::vector<size_t>& offsets, std::vector<char>& output,
    char* regexp);

//__shared__ char auto_data[];
//__device__ Reprog* automaton;

/**
 * @fn  __global__ void addKernel(char *input, const size_t *offsets, void* memory,
 * Reljunk* dev_rj, Resub* dev_rs, size_t mem_diff, size_t mem_size, size_t reinst_size,
 * char *output, size_t size)
 *
 * @brief   Kernel function. Starts matching for each passed string.
 *
 * @param [in]  input       Input file, concatenated with \0 symbol
 * @param   offsets         The offsets of lines, position of first symbol in each string.
 * @param [in]  memory      Regexp programs data. Number of programs equals to number of 
 * 				            input lines in file
 * @param [in]  dev_rj      Preallocated Reljunk array. Temp data.
 * @param [in]  dev_rs      Preallocated Resub array. Results data.
 * @param   mem_diff        The memory address difference between automaton in 
 * 							PC memory and CUDA memory. In bytes.
 * @param   mem_size        Size of one regexp automaton program.
 * @param   reinst_size     Size of the reinst for relocation.
 * @param [in,out]  output  Output boolean array [string match/not match].
 * @param   size            The count of lines in input file.
 */
__global__ void addKernel(char *input, 
    const size_t *offsets, 
    void* memory, 
    Reljunk* dev_rj, 
    Resub* dev_rs, 
    size_t mem_diff, 
    size_t mem_size, 
    size_t reinst_size, 
    char *output, 
    size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Reprog* p;
    Reljunk* rj;
    Resub* rs;

    // Experiments
    //if(i == blockIdx.x * blockDim.x) {
    //    for(int j = 0; j < mem_size; ++j) {
    //        auto_data[j] = *((char*)memory + j);
    //    }
    //    automaton = (Reprog*) &auto_data[0];
    //    output[0] = automaton->firstinst->type;
    //}

    if(i < size) {
        // Preparations for each thread.
        p = (Reprog*) ((char*)memory + i*mem_size);
        p = cu_relocate(p, mem_diff+i*mem_size, reinst_size);
        rj = (Reljunk*) ((char*)dev_rj + i*sizeof(Reljunk));
        rs = (Resub*) ((char*)dev_rs + i*sizeof(Resub));

        rs->s.sp = 0;
        rs->e.ep = 0;
        output[i] = regexec9(p, input + offsets[i], rj, rs, 1, i);
    }
}

int main()
{
    char* regexp = "([0-9][0-9]?)/([0-9][0-9]?)/([0-9][0-9]([0-9][0-9])?)";
    //std::cin >> regexp;

    std::vector<char> input;
    std::vector<size_t> offsets;
    std::vector<char> line;

    std::ifstream inputFile("input.txt"); 
    offsets.push_back(0);

    /**
    * Reading file line-by-line and puts chars it vector, dividing 
    * each string with \0 character.
    * 
    * Also filling offsets vector.
    */
    while(inputFile) {
        char c;
        inputFile.get(c);
        if(c == '\n') {
            input.push_back(0);
            if(!inputFile.eof())
                offsets.push_back(input.size());
        } else {
            input.push_back(c);
        }
    }
    input.push_back(0);

    std::vector<char> output(offsets.size(), 0);

    // Starting the matching
    matchWithCuda(input, offsets, output, regexp);

    // Showing results to user
    for (int i = 0; i < output.size(); i++) {
        if(output[i])
            std::cout << "Line " << i+1 << ":" << (int) output[i] << std::endl;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
    checkCudaErrors(cudaDeviceReset());

    return 0;
}

/**
 * @fn  void matchWithCuda(const std::vector<char>& input, const std::vector<size_t>& offsets,
 * std::vector<char>& output, char* regexp)
 *
 * @brief   Prepares kernel data and runs it.
 *
 * @param   input           The input char vector.
 * @param   offsets         The offsets vector.
 * @param [in,out]  output  The output.
 * @param [in]  regexp      Regular expression.
 */
void matchWithCuda(const std::vector<char>& input, 
    const std::vector<size_t>& offsets, std::vector<char>& output, char* regexp)
{
    char* dev_input = 0;
    size_t* dev_offsets = 0;
    char* dev_output = 0;
    void* dev_memory = 0;

    // Preparing regexp.
    size_t memory_size = sizeof(Reprog) + 6*sizeof(Reinst)*strlen(regexp);
    void* memory = malloc(memory_size);
    // Compiling regexp.
    Reprog* p = cu_regcomp9(regexp, memory);

    // Allocate space for automatons
    checkCudaErrors(cudaMalloc((void**)&dev_memory, memory_size*offsets.size()));

    // Additional information for relocation.
    size_t mem_diff = (char*) dev_memory - (char*) memory;
    size_t reinst_size = cu_get_reinst_size((char*)memory);

    // Copying automata to device
    for(int i = 0; i < offsets.size(); i++) {
        checkCudaErrors(cudaMemcpy((void*)((char*)dev_memory+i*memory_size), memory, memory_size, cudaMemcpyHostToDevice));
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    // Experiments
    // checkCudaErrors(cudaMemcpyToSymbol((char*)automaton, dev_memory, memory_size));

    // Allocate memory on device
    checkCudaErrors(cudaMalloc((void**)&dev_input, 
        input.size() * sizeof(char)));

    checkCudaErrors(cudaMalloc((void**)&dev_offsets, offsets.size() * sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&dev_output, offsets.size() * sizeof(char)));

    Reljunk* dev_rj;
    checkCudaErrors(cudaMalloc((void**)&dev_rj, offsets.size() * sizeof(Reljunk)));

    Resub* dev_rs;
    checkCudaErrors(cudaMalloc((void**)&dev_rs, offsets.size() * sizeof(Resub)));

    //Relist* dev_relist0;
    //checkCudaErrors(cudaMalloc((void**)&dev_relist0, offsets.size() * LISTSIZE * sizeof(Relist)));

    //Relist* dev_relist1;
    //checkCudaErrors(cudaMalloc((void**)&dev_relist1, offsets.size() * LISTSIZE * sizeof(Relist)));

    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(dev_input, &input[0],
        input.size() * sizeof(char), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(dev_offsets, &offsets[0],
        offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    // Create timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Starting timer
    cudaEventRecord(start, 0);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<BLOCKS, THREADS>>>(dev_input, dev_offsets, dev_memory, dev_rj, dev_rs,
        mem_diff, memory_size, reinst_size, dev_output, offsets.size());

    // Stopping timer and showing information
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for the GPU: %f ms\n", time);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCudaErrors(cudaMemcpy(&output[0], dev_output,
        offsets.size() * sizeof(char), cudaMemcpyDeviceToHost));

    cudaFree(dev_memory);
    cudaFree(dev_input);
    cudaFree(dev_offsets);
    cudaFree(dev_output);    
}