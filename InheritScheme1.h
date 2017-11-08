#ifndef EIGENTENSORS_INHERITSCHEME1_H
#define EIGENTENSORS_INHERITSCHEME1_H

#include "ConstScheme2.h"

template< typename Scalar >
struct InheritScheme1 : public ConstScheme3<Scalar>
{
    typedef ConstScheme3<Scalar> MyBase;
    //using MyBase::MyBase;

    // Without explicit, this constructor can call itself
    // if it does not match any in parent class
    //
    // Using 'using' converts all MyBase arguments into this
    // class, disabling inheritance

    template< typename ... Args >
    explicit InheritScheme1( Args ... args )
     : MyBase( args... )
    {}
};

#endif //EIGENTENSORS_INHERITSCHEME1_H
