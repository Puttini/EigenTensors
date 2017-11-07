#ifndef EIGENTENSORS_CONSTSCHEME1_H
#define EIGENTENSORS_CONSTSCHEME1_H

template< typename Scalar >
struct ConstScheme1
{
    Scalar* i;

    ConstScheme1( Scalar* i ) : i(i)
    {}

    ConstScheme1( ConstScheme1& o ) : i(o.i)
    {}

    Scalar f() { return *i; }
    Scalar& f() const { return *i; }
};

template< typename Scalar >
struct ConstScheme1< const Scalar >
{
    const Scalar* i;

    ConstScheme1( const Scalar* i ) : i(i)
    {}

    ConstScheme1( const ConstScheme1<const Scalar>& o ) : i(o.i)
    {}

    ConstScheme1( const ConstScheme1<Scalar>& o )
     : ConstScheme1( *reinterpret_cast<const ConstScheme1<const Scalar>*>(&o) )
    {}

    Scalar f() const { return *i; }
};

#endif //EIGENTENSORS_CONSTSCHEME1_H
