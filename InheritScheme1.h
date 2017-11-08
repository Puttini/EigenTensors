#ifndef EIGENTENSORS_INHERITSCHEME1_H
#define EIGENTENSORS_INHERITSCHEME1_H

#include "ConstScheme2.h"

template< typename Scalar >
struct InheritScheme1 : public ConstScheme3<Scalar>
{
    typedef ConstScheme3<Scalar> MyBase;
    using MyBase::MyBase;

    // Can cause infinite recursive loop using copy constructor ?
    template< typename ... Args >
    InheritScheme1( Args ... args )
     : MyBase( args... )
    {}
};

#endif //EIGENTENSORS_INHERITSCHEME1_H
